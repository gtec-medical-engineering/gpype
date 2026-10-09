"""Season 4, Episode 9: a recording handed to MNE-Python, in volts.

Records four named channels and a marker channel, with their units, to
HDF5 and to CSV. The HDF5 copy becomes an mne.io.RawArray: signal
channels as EEG in volts, the trigger as a stim channel, its markers as
annotations. The CSV copy stores no units, and the bridge refuses it
rather than guess microvolts. Requires mne: pip install mne
"""

import glob
import os
import tempfile
import time

import mne
import numpy as np

import gpype as gp

RATE = 250
OUT = tempfile.gettempdir()
LABELS = ["Fz", "C3", "Cz", "C4", "marker"]
#: A Generator measures nothing, so say what it stands for. A marker
#: channel carries codes, not a physical quantity, so it has no unit.
UNITS = ["uV", "uV", "uV", "uV", None]
#: What each role becomes in MNE. Any other role is refused.
MNE_TYPES = {"signal": "eeg", "trigger": "stim"}


def newest(extension: str) -> str:
    """The file a writer produced last; writers timestamp their names."""
    pattern = os.path.join(OUT, f"s4e9_*{extension}")
    return sorted(glob.glob(pattern))[-1]


def record() -> None:
    """Record under three seconds and three markers, as Episode 7 does."""
    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=RATE, channel_count=4,
                          signal_frequency=10, signal_amplitude=20.0,
                          noise_amplitude=2.0)
    marker = gp.Marker()
    both = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])
    described = gp.ChannelLabeler(labels=LABELS, units=UNITS)
    p.connect(source, both["in1"])
    p.connect(marker, both["in2"])
    p.connect(both, described)
    stem = os.path.join(OUT, "s4e9")
    p.connect(described, gp.HDF5Writer(file_name=stem + ".h5"))
    p.connect(described, gp.CsvWriter(file_name=stem + ".csv"))
    p.start()
    time.sleep(0.8)
    for code in (1, 2, 1):
        marker.emit(code)
        time.sleep(0.6)
    p.stop()
    p.close()


def read(reader) -> gp.Result:
    """Read a whole recording in one batch run."""
    with gp.Pipeline() as p:
        p.connect(reader, gp.Collector())
        return p.run()


def to_mne(result: gp.Result) -> mne.io.RawArray:
    """Build an MNE Raw from a Result, or refuse.

    Signal channels are scaled to volts by their own unit; trigger codes
    pass through unscaled, as MNE expects of a stim channel.

    Raises:
        ValueError: If the result records no units, if a signal channel's
            unit is not one g.Pype can scale, or if a channel has a role
            MNE_TYPES does not map.
    """
    if result.si_scale is None:
        raise ValueError(
            "this recording records no units, so its scale is unknown. "
            "A missing unit is not taken to be microvolts: record with "
            "ChannelLabeler(units=...) or an amplifier source, to a "
            "format that keeps them."
        )
    types, factors = [], []
    for channel in result.channels:
        if channel.role not in MNE_TYPES:
            raise ValueError(
                f"channel {channel.name!r} has role {channel.role!r}; "
                f"select the signal and trigger channels first."
            )
        types.append(MNE_TYPES[channel.role])
        if channel.role == "trigger":
            factors.append(1.0)
        elif channel.si_scale is None:
            raise ValueError(
                f"channel {channel.name!r} has unit {channel.unit!r}, "
                f"which g.Pype cannot put in volts."
            )
        else:
            factors.append(channel.si_scale)

    data = result.data.T * np.asarray(factors)[:, None]
    info = mne.create_info(result.labels, result.rate, types)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(mne.Annotations(
        onset=[event.sample / result.rate for event in result.events],
        duration=[event.duration / result.rate for event in result.events],
        description=[event.label for event in result.events],
    ))
    return raw


if __name__ == "__main__":

    record()

    # 1. A recording that kept its units
    result = read(gp.HDF5Reader(file_name=newest(".h5"), mode="batch",
                                variable_name="data", has_time_row=True))
    print("units    :", result.units)
    print("si_scale :", result.si_scale)

    raw = to_mne(result)
    print()
    print(raw)
    print("types    :", raw.get_channel_types())
    notes = raw.annotations
    print("markers  :", dict(zip(notes.onset.round(3).tolist(),
                                 notes.description.tolist())))
    print("Cz, first samples in volts:", raw.get_data(picks="Cz")[0, :3])

    # 2. One that did not: a CSV stores no units and no roles, so its
    # Result says so, and the bridge refuses it instead of guessing
    print()
    try:
        to_mne(read(gp.CsvReader(file_name=newest(".csv"), mode="batch")))
    except ValueError as error:
        print("ValueError:", error)
