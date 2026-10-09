"""Filling in what a pipeline document leaves implicit.

A document that omits node ids works perfectly in one process and fails
in a way that is hard to see across two. Both halves of a distributed
pipeline derive a stream id from each node's document id, and a node
without one is given a *fresh random* id by whichever process built it --
so the edge and the server end up naming the same node differently, every
frame is routed to a stream nobody is receiving, and the broker discards
it. Measured 2026-08-30, edge to container:

    Discarding a frame for stream '1AF878549BE87C6', which nothing is
    receiving. Registered streams: ['B65B7B8034F29578'].

Nothing is broken at either end. The two are reading what is, in the only
way that matters here, two different documents.

**Why the ids are not simply derived instead.** Because every derivation
from the document's own content collides. ``chain_params`` already
rejected deriving from the name, in writing -- *"two unnamed chains of the
same class both take the class name, verified, so deriving from the name
would pair two sinks on the same stream and the broker's last-writer-wins
would cross them silently"* -- and since ioiocore defaults a node's name
to its class name, deriving from the class is that same collision. Position
in the list moves when a node is inserted; a hash of the configuration
collides for two identical siblings, which is exactly the parallel-branch
case a pipeline is most likely to contain.

So the id stays what it is -- explicit, persisted state -- and this module
removes the only real objection to that, which is having to invent one by
hand. Assign the ids **once**, keep them, and let both halves read the
result.

Documents produced by ``serialize()`` already carry ids and need none of
this.
"""

from __future__ import annotations

import copy

#: Where a node's identity lives inside its ``config``.
ID_KEY = "id"

#: ioiocore's placeholder for "generate one for me" -- treated here as
#: equivalent to absent, because a document that says TBG is asking for
#: exactly the thing this function does, and asking two processes
#: separately is what causes the mismatch.
UNASSIGNED = "TBG"

#: Where a node's port configurations live inside its ``config``.
PORT_KEYS = ("input_ports", "output_ports")

#: Where a chain that does not derive its internals
#: (``INTERNALS_ARE_DERIVED = False``) writes them, ids included, inside
#: its ``config``: ioiocore's ``Chain.serialize``.
INTERNAL_NODES_KEY = "_internal_nodes"


def _needs_id(config: dict) -> bool:
    """Whether this node's configuration lacks a usable id."""
    value = config.get(ID_KEY)
    return not isinstance(value, str) or not value or value == UNASSIGNED


def canonicalize(document: dict) -> dict:
    """Return *document* with every node's id filled in.

    Call this once, on the authoring side, and give the result to both
    halves of a distributed pipeline -- or store it, which is the same
    thing done earlier. The ids are minted the way ioiocore mints them,
    so a canonicalized document is indistinguishable from one that came
    out of ``serialize()``.

    Idempotent: a node that already has an id keeps it, so canonicalizing
    twice changes nothing and canonicalizing a document that gained a new
    node touches only the new node. That is the property that makes this
    safe to run in a save path.

    The input is not modified. A caller holding a document it intends to
    reuse should not find it silently altered.

    Args:
        document: A pipeline document, as loaded from JSON.

    Returns:
        dict: A copy with an id on every node.

    Raises:
        TypeError: If *document* is not a mapping.

    Note:
        Malformed nodes are left alone rather than repaired. Deciding
        what a node with no ``module`` means is the loader's job, and it
        already refuses such a document with a message naming the field;
        a second opinion here would only produce a different error for
        the same mistake.
    """
    if not isinstance(document, dict):
        raise TypeError(
            f"a pipeline document is a mapping, not {type(document).__name__}"
        )

    result = copy.deepcopy(document)
    nodes = result.get("nodes")
    if not isinstance(nodes, list):
        return result

    from ioiocore.imp.portable_imp import PortableImp

    for node in nodes:
        if not isinstance(node, dict):
            continue
        config = node.get("config")
        if not isinstance(config, dict):
            config = {}
            node["config"] = config
        if _needs_id(config):
            config[ID_KEY] = PortableImp.new_id()

    return result


def with_fresh_ids(document: dict) -> dict:
    """Return *document* with a new id on every node and every port.

    For building one document into several live pipelines in one
    process. ioiocore's id registry is process-wide, and it refuses an
    object whose id a live one holds, so deserializing the same document
    twice raises ``ID ... conflict``. ``Portable.reset()`` would clear
    the registry under every pipeline already built (D-BATCH-72).

    A connection that names a port by id is rewritten to that port's new
    id. One written by name, ``"eeg.out"`` or ``"eeg"``, is kept: names
    do not change. A node or port without an id gets one. An explicit
    ``stream_id`` is kept. The internal nodes a chain writes into its
    configuration (``INTERNALS_ARE_DERIVED = False``) are renewed the
    same way, and a connection naming one of their ports follows it.

    Not for the halves of a distributed pipeline: a stream id derived
    from a node's id would change on one side only. Give both halves one
    document, as :func:`canonicalize` says.

    The input is not modified.

    Args:
        document: A pipeline document, as ``serialize()`` wrote it or as
            loaded from JSON.

    Returns:
        dict: A copy with fresh ids and its connections rewritten.

    Raises:
        TypeError: If *document* is not a mapping.

    Note:
        As with :func:`canonicalize`, malformed entries are left for the
        loader to refuse. An id repeated in the input stays repeated, so
        the loader still refuses that document.
    """
    if not isinstance(document, dict):
        raise TypeError(
            f"a pipeline document is a mapping, not {type(document).__name__}"
        )

    result = copy.deepcopy(document)
    nodes = result.get("nodes")
    if not isinstance(nodes, list):
        return result

    from ioiocore.imp.portable_imp import PortableImp

    # Old id -> new id, so every reference to one id follows it.
    minted: dict = {}

    def fresh(old) -> str:
        if not isinstance(old, str) or not old or old == UNASSIGNED:
            return PortableImp.new_id()
        if old not in minted:
            minted[old] = PortableImp.new_id()
        return minted[old]

    def renew(node: dict) -> dict:
        # Plain dicts, as a document loaded from JSON has: `serialize()`
        # hands out ioiocore's read-only Configuration objects, and a
        # deep copy keeps their type.
        config = node.get("config")
        config = dict(config) if isinstance(config, dict) else {}
        node["config"] = config
        config[ID_KEY] = fresh(config.get(ID_KEY))
        for key in PORT_KEYS:
            ports = config.get(key)
            if not isinstance(ports, list):
                continue
            config[key] = [
                (
                    {**port, ID_KEY: fresh(port.get(ID_KEY))}
                    if isinstance(port, dict)
                    else port
                )
                for port in ports
            ]
        internal = config.get(INTERNAL_NODES_KEY)
        if isinstance(internal, list):
            config[INTERNAL_NODES_KEY] = [
                renew(dict(inner)) if isinstance(inner, dict) else inner
                for inner in internal
            ]
        return node

    for node in nodes:
        if isinstance(node, dict):
            renew(node)

    connections = result.get("connections")
    if isinstance(connections, list):
        result["connections"] = [
            (
                [
                    minted.get(end, end) if isinstance(end, str) else end
                    for end in pair
                ]
                if isinstance(pair, (list, tuple))
                else pair
            )
            for pair in connections
        ]
    return result


def omit_unassigned_edge(serialized: dict) -> dict:
    """Write a world-facing node that names no edge as it was before.

    A node's configuration carries ``edge_id`` even when it is None, so
    that the node describes itself completely. A document does not: an
    unassigned node is written with no such key at all, which keeps
    every document that never assigned one byte-identical to a 4.0
    document, and keeps the key's presence meaning that somebody chose.

    Args:
        serialized: One node as ``serialize()`` produced it.

    Returns:
        The same dict, its ``config`` replaced by a copy without the key
        when the id was None; untouched otherwise.
    """
    from .constants import Constants

    key = Constants.Keys.EDGE_ID
    config = serialized.get("config")
    if isinstance(config, dict) and key in config and config[key] is None:
        serialized["config"] = {k: v for k, v in config.items() if k != key}
    return serialized


def node_ids(document: dict) -> list:
    """The id of every node in *document*, in order.

    For checking that two halves really are reading the same document --
    the failure this module exists to prevent is silent, so being able to
    compare the two lists is worth the six lines.

    Args:
        document: A pipeline document.

    Returns:
        list: One entry per node; None where a node carries no id.
    """
    nodes = document.get("nodes") if isinstance(document, dict) else None
    if not isinstance(nodes, list):
        return []
    out = []
    for node in nodes:
        config = node.get("config") if isinstance(node, dict) else None
        value = config.get(ID_KEY) if isinstance(config, dict) else None
        out.append(value if isinstance(value, str) and value else None)
    return out
