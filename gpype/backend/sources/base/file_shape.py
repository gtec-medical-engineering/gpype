"""A reader takes its shape from its file, and says so when it disagrees.

``GtcReader`` and the four recording readers read their sampling rate,
their channel count and, in batch mode, their frame size from the file.
The same three also arrive as keyword arguments: from an author by
hand, or from a stored configuration beside the parameters that decide
them. One that agrees with the file is used. One that disagrees is
refused from an author, and warned about from a document, whose file
changed after it was saved (D-NODE-61, D-BATCH-75).
"""

from __future__ import annotations

import os
import sys
import warnings

import ioiocore as ioc

from ....common.constants import Constants
from ...core._private import assembly
from .source import ABSENT, Source

#: The three settings a file decides.
SHAPE_KEYS = (
    Constants.Keys.SAMPLING_RATE,
    Constants.Keys.CHANNEL_COUNT,
    Constants.Keys.FRAME_SIZE,
)

#: How far a given rate may sit from one a recorded time column gives,
#: relative, and still agree: the threshold ``rate_from_time_column``
#: snaps to an integer at. A time column states its rate only to its
#: text's precision, so exact equality would refuse a correct 500/3. A
#: rate the file states, in ``gpype_meta``, an HDF5 attribute or the EDF
#: header, is compared exactly.
TIME_COLUMN_TOLERANCE = 1e-4


def _folder(path: str) -> str:
    """Return the normalised folder holding *path*."""
    return os.path.normcase(os.path.dirname(os.path.abspath(path)))


def loader_stacklevel() -> int:
    """Return the stacklevel that points a warning at who loaded a document.

    For a ``warnings.warn`` made by the function that calls this one, and
    counted the way ``warnings.warn`` counts. ``Pipeline.deserialize``
    constructs a node from inside ioiocore, at a depth that depends on
    which of ioiocore's modules were compiled, so a fixed stacklevel
    lands in ``ioiocore/portable.py``. This walks up instead, to the
    first frame above ``Pipeline.deserialize`` that is neither g.Pype's
    nor ioiocore's.

    Returns:
        That frame's stacklevel. Where ``Pipeline.deserialize`` is not on
        the stack, the first such frame above the caller's; where there
        is none, the caller's caller.
    """
    from ...pipeline import Pipeline

    # g.Pype's package folder is three above this module's.
    here = _folder(__file__)
    package = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    packages = (package + os.sep, _folder(ioc.__file__) + os.sep)
    deserialize = Pipeline.deserialize.__code__

    # frames[0] is the caller, which is stacklevel 1.
    frames = []
    frame = sys._getframe(1)
    while frame is not None:
        frames.append(frame)
        frame = frame.f_back
    first = next(
        (i + 1 for i, f in enumerate(frames) if f.f_code is deserialize), 1
    )
    for index in range(first, len(frames)):
        name = os.path.normcase(
            os.path.abspath(frames[index].f_code.co_filename)
        )
        if not name.startswith(packages):
            return index + 1
    return 2


def _agrees(key: str, value, derived, tolerance: float) -> bool:
    """Whether a given value describes what the file decides."""
    try:
        value, derived = float(value), float(derived)
    except (TypeError, ValueError):
        return False
    if key == Constants.Keys.SAMPLING_RATE and tolerance:
        return abs(value - derived) <= tolerance * abs(derived)
    return value == derived


def resolve(
    file_name: str,
    given: dict,
    derived: dict,
    node: str,
    advice: str,
    rate_tolerance: float = 0.0,
) -> dict:
    """Return the shape to build with, refusing or warning on disagreement.

    Args:
        file_name: Path to the recording, for the messages.
        given: What the reader was given per key of ``derived``:
            ``ABSENT`` or None where nothing.
        derived: What the file decides per key, None where it decides
            nothing. Only these keys are checked.
        node: The reader as a message names it.
        advice: What the file decides, and what to drop, for the
            refusal.
        rate_tolerance: Relative difference at which a given rate still
            agrees. Zero asks for the file's rate exactly.

    Returns:
        The value per key: the given one where it agrees or the file
        decides nothing, the file's otherwise.

    Raises:
        ValueError: If a value its author gave disagrees with the file.
            A document's is warned about instead, with a
            ``RuntimeWarning`` that points at the line that loaded the
            document.
    """
    chosen = {}
    differing = []
    for key, decided in derived.items():
        value = Source.scalar(given.get(key, ABSENT))
        absent = value is ABSENT or value is None
        if decided is None:
            chosen[key] = None if absent else value
        elif absent:
            chosen[key] = decided
        elif _agrees(key, value, decided, rate_tolerance):
            chosen[key] = value
        else:
            chosen[key] = decided
            differing.append(
                f"{key}={value!r}, where the file gives {decided}"
            )
    if not differing:
        return chosen

    found = "; ".join(differing)
    if assembly.is_rebuilding():
        warnings.warn(
            f"The document's {node} disagrees with '{file_name}': "
            f"{found}. The file has changed since the document was "
            f"written, or the document was edited; the file's shape is "
            f"used, and saving the document again writes it.",
            RuntimeWarning,
            stacklevel=loader_stacklevel(),
        )
        return chosen
    raise ValueError(
        f"{node} takes its shape from '{file_name}', and was given "
        f"{found}. {advice}"
    )


def recording(
    reader,
    file_name: str,
    kwargs: dict,
    sampling_rate,
    frame_size,
    batch: bool,
    file_rate,
    samples,
    rate_from_times: bool,
) -> tuple:
    """Resolve a recording reader's shape against the file it read.

    For ``CsvReader`` and the ``RecordingReader`` family. The channel
    count is the file's column count. The rate is the file's where the
    file records one. A given rate agrees with one derived from a time
    column to ``TIME_COLUMN_TOLERANCE``, and with one the file states
    exactly. The frame size is checked in batch mode only, where it is
    the sample count; there the default, one, is no author's word. In
    realtime mode it is how many samples a cycle emits, which the file
    does not decide.

    Args:
        reader: The node being built.
        file_name: Path to the recording.
        kwargs: The reader's keyword arguments. The given channel count
            is taken out, and the one to build with put back.
        sampling_rate: The ``sampling_rate`` argument, or None.
        frame_size: The ``frame_size`` argument, a list unwrapped.
        batch: Whether the reader runs in batch mode.
        file_rate: The rate the file records, or None.
        samples: The recording, ``(time, channel)``.
        rate_from_times: Whether ``file_rate`` was derived from a time
            column rather than stated by the file.

    Returns:
        ``(sampling_rate, frame_size)`` to build with. The rate is None
        where neither the file nor the caller gives one.

    Raises:
        ValueError: If a value its author gave disagrees with the file.
    """
    keys = Constants.Keys
    given = {
        keys.SAMPLING_RATE: sampling_rate,
        keys.CHANNEL_COUNT: kwargs.pop(keys.CHANNEL_COUNT, ABSENT),
    }
    derived = {
        keys.SAMPLING_RATE: file_rate,
        keys.CHANNEL_COUNT: int(samples.shape[1]),
    }
    if batch:
        given[keys.FRAME_SIZE] = ABSENT if frame_size == 1 else frame_size
        derived[keys.FRAME_SIZE] = int(samples.shape[0])
    shape = resolve(
        file_name,
        given,
        derived,
        node_name(reader, kwargs),
        "The rate is the file's where it records one, the channel count "
        "is the file's, and in batch mode the frame size is its sample "
        "count: drop what disagrees.",
        rate_tolerance=TIME_COLUMN_TOLERANCE if rate_from_times else 0.0,
    )
    kwargs[keys.CHANNEL_COUNT] = shape[keys.CHANNEL_COUNT]
    if batch:
        frame_size = shape[keys.FRAME_SIZE]
    return shape[keys.SAMPLING_RATE], frame_size


def node_name(reader, kwargs: dict) -> str:
    """Return the reader as a message names it.

    Args:
        reader: The node being built.
        kwargs: Its keyword arguments. A document stores every node's
            name, so this is the one it knows the reader by; unnamed,
            that is the class's.

    Returns:
        ``"Class"`` or ``"Class 'name'"``.
    """
    name = kwargs.get("name")
    cls = type(reader).__name__
    return cls if name in (None, cls) else f"{cls} '{name}'"
