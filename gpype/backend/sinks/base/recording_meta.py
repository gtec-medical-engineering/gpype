"""What a g.Pype recording carries beside its samples, described once.

MAT, HDF5 and EDF each have their own place to stash a document, and the
three untracked drafts this module replaces each used it differently --
which is why none of them could read a file either of the others wrote,
and why HDF5Writer's draft could not even read its own: it recorded
nothing at all. MatWriter's draft flattened the port context with
``str(value)``, so the channel labels ended up on disk as the Python repr
``"['Fz', 'Cz', 'Pz']"`` -- measured -- and getting them back would need
``ast.literal_eval`` on text that came out of a file, not a description a
reader can trust.

One JSON document, built and parsed by exactly the functions below, is
what lets :mod:`mat_writer`, :mod:`hdf5_writer` and their readers agree on
one shape rather than three. EDF has no place to put an arbitrary
document -- its per-signal header fields are fixed and short, and its
annotation channel has room for roughly one entry per data record
(measured: 64 written into a 10-record file, 10 recovered) -- so
:mod:`edf_writer` does not use this module for the bulk of its metadata;
it uses the format's own fields (``transducer`` for role, ``equipment``
for the device serial, ``recording_additional`` for the mark) and reaches
here only indirectly, for the shared vocabulary of key names.
"""

from __future__ import annotations

import json
from typing import Optional

from ....common._private import channels
from ....common._private.entitlement import MARK
from ....common.constants import Constants

#: Version tag stored in every metadata document, and the prefix a reader
#: accepts. Pinned to the major version only ("1", not "1.0") so a future
#: field can be added without every existing reader needing a bump: this
#: is the same reasoning API_VERSION documents for the control surface.
META_VERSION = "gpype-recording/1"

#: Name of the dataset (HDF5/MAT) that carries the metadata document.
#: Shared between :mod:`mat_writer` and :mod:`hdf5_writer` so a reader
#: written against one recognises a file produced by the other.
META_DATASET = "gpype_meta"


def collect(
    port_context_in: dict, marked: bool, sampling_rate: Optional[float]
) -> dict:
    """Build the metadata document for a recording that is starting.

    Called from ``_open_file``, before any sample has arrived, so the
    document a killed recording leaves behind still names its rate and
    its channels -- unlike write-at-close, which was measured to leave a
    MATLAB file MATLAB will not load if the process dies mid-recording.

    Args:
        port_context_in: The writer's input port contexts, as passed to
            ``_open_file``.
        marked: Whether this run's artifacts must carry the entitlement
            mark, i.e. the writer's own ``_marked``.
        sampling_rate: The rate ``FileWriter.setup`` read from the port
            context. Passed explicitly rather than re-read from the
            context here, so the one number every writer already
            validated is the one that gets recorded. None omits the key
            entirely, which is what a *sparse* stream needs: an event
            source publishes no rate, and recording a zero would make a
            reader take it for a continuous stream sampled at nothing.

    Returns:
        A JSON-safe dict. Keys that the context does not carry are
        absent rather than defaulted -- ``channel_labels`` in particular:
        an unlabelled input must stay unlabelled, because inventing
        positional names here would make a reader believe a montage said
        so when nothing did (the same rule ``channels.describe`` enforces
        for a live port context, applied to what gets written to disk).
    """
    context = (port_context_in or {}).get(Constants.Defaults.PORT_IN) or {}
    meta: dict = {"format": META_VERSION}
    if sampling_rate is not None:
        meta["sampling_rate"] = float(sampling_rate)

    count = context.get(Constants.Keys.CHANNEL_COUNT)
    if count is not None:
        meta["channel_count"] = int(count)

    # Guarded by has_labels/has_roles, exactly as csv_writer.py does for
    # its header line: a channel that was never named must not become
    # "Ch01" on disk, because a reader loading it back would then have no
    # way to tell an invented name from a real one.
    if channels.has_labels(context):
        meta["channel_labels"] = channels.labels_of(context)
    if channels.has_roles(context):
        meta["channel_roles"] = channels.roles_of(context)

    exact = context.get(Constants.Keys.SAMPLING_RATE_EXACT)
    if exact is not None:
        meta["sampling_rate_exact"] = [int(exact[0]), int(exact[1])]

    units = context.get(Constants.Keys.CHANNEL_UNITS)
    if units is not None:
        meta["channel_units"] = list(units)

    serial = context.get(Constants.Keys.DEVICE_SERIAL)
    if serial:
        meta["device_serial"] = str(serial)

    system = channels.montage_system(context)
    if system is not None:
        meta["montage_system"] = system

    start_time = context.get(Constants.Keys.START_TIME)
    if start_time is not None:
        meta["start_time"] = str(start_time)

    if marked:
        meta["mark"] = MARK

    return meta


def finalize(meta: dict, sample_count: int) -> dict:
    """Return *meta* with the final sample count recorded.

    Called from ``_close_file``, once the total is known. A shallow copy
    rather than a mutation, so a caller holding the document written at
    open time (which is what a killed recording leaves on disk) is not
    surprised by a count it never got to write.

    Args:
        meta: The document :func:`collect` built.
        sample_count: Total samples actually written.

    Returns:
        A new dict, identical to *meta* plus ``sample_count``.
    """
    out = dict(meta)
    out["sample_count"] = int(sample_count)
    return out


def parse(blob: str) -> dict:
    """Parse a metadata document read back from a file.

    Args:
        blob: The JSON text, exactly as ``collect``/``finalize`` produced
            it -- ``str`` even for the MAT encoding, which stores it as a
            column of ``uint16`` character codes; the reader decodes that
            to a Python string before calling here.

    Returns:
        The parsed document.

    Raises:
        ValueError: If the document does not declare a version this
            reader understands.
    """
    meta = json.loads(blob)
    version = str(meta.get("format", ""))
    if not version.startswith(META_VERSION):
        raise ValueError(
            f"this file's recording metadata is version {version!r}, "
            f"which this reader does not recognise; it understands "
            f"'{META_VERSION}.x'."
        )
    return meta
