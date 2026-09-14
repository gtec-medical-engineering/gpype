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
