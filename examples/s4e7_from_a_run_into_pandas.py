"""Season 4, Episode 7: one recording read by hand and through a Result.

Requires pandas, which is not a g.Pype dependency: pip install pandas
"""

import glob
import os
import tempfile
import time

import numpy as np
import pandas as pd

import gpype as gp

SAMPLING_RATE = 250
OUT = tempfile.gettempdir()
LABELS = ["Fz", "C3", "Cz", "C4", "marker"]


def newest(stem: str, extension: str) -> str:
    """The file a writer produced last, timestamp and all."""
    found = sorted(glob.glob(os.path.join(OUT, f"{stem}_*{extension}")))
    return found[-1] if found else ""


def record() -> None:
    """Record four channels and a marker channel to CSV and HDF5."""
    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=4,
                          signal_frequency=10, signal_amplitude=20.0,
                          noise_amplitude=2.0)
    # Markers fired by this script, standing in for a paradigm
    marker = gp.Marker()
    both = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])
    names = gp.ChannelLabeler(labels=LABELS)
    p.connect(source, both["in1"])
    p.connect(marker, both["in2"])
    p.connect(both, names)
    p.connect(names, gp.CsvWriter(file_name=os.path.join(OUT, "s4e7.csv")))
    p.connect(names, gp.HDF5Writer(file_name=os.path.join(OUT, "s4e7.h5")))
    p.start()
    time.sleep(0.8)
    for code in (1, 2, 1):
        marker.emit(code)
        time.sleep(0.5)
    time.sleep(0.2)
    p.stop()
    p.close()


def read(reader) -> gp.Result:
    """Read a whole recording in one batch run."""
    with gp.Pipeline() as p:
        p.connect(reader, gp.Collector())
        return p.run()


if __name__ == "__main__":

    record()
    csv = newest("s4e7", ".csv")

    # 1. By hand. The file opens with comment lines, its header separates
    # names with ", ", pandas' default float parser can misread the 17th
    # digit g.Pype writes, and the time only becomes the index if you
    # make it one
    by_hand = pd.read_csv(csv, comment="#", skipinitialspace=True,
                          float_precision="round_trip")
    by_hand = by_hand.set_index("Time")

    # 2. Through a Result: the reader already knows the columns, the
    # rate and the time axis
    result = read(gp.CsvReader(file_name=csv, mode="batch"))
    from_result = result.to_dataframe()

    print("pd.read_csv, by hand:")
    print(by_hand.head(3))
    print()
    print("result.to_dataframe():")
    print(from_result.head(3))
    print()
    print("  same columns :", list(by_hand.columns) == list(from_result))
    print("  same values  :", np.array_equal(by_hand.to_numpy(),
                                             from_result.to_numpy()))
    print("  same times   :", np.array_equal(by_hand.index.to_numpy(),
                                             from_result.index.to_numpy()))

    # A DataFrame answers questions a flat array does not: every row a
    # marker is on, with its time
    fired = from_result[from_result["marker"] != 0]
    print()
    print("markers:", {t: int(v) for t, v in fired["marker"].items()})

    # 3. What the frame cannot say: which column is the trigger. A CSV
    # stores no roles, so its Result calls every column a signal; the
    # HDF5 copy of the same recording kept them
    twin = read(gp.HDF5Reader(file_name=newest("s4e7", ".h5"),
                              mode="batch", variable_name="data",
                              has_time_row=True))
    print()
    print("result.roles, from the .csv :", result.roles)
    print("result.roles, from the .h5  :", twin.roles)
    print("the trigger column          :",
          [c.name for c in twin.channels if c.role == "trigger"])
    print("result.times[:3]            :", twin.times[:3])
    print("result.channels[-1]         :", twin.channels[-1])
