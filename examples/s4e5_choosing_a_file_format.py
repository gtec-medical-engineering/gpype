"""Season 4, Episode 5: one recording written four ways, and read back.

The files are left in the temporary folder, as s4e5_<date>_<time>.*
"""

import glob
import os
import tempfile
import time

import numpy as np

import gpype as gp

SAMPLING_RATE = 250
SECONDS = 4.0
ELECTRODES = ["Fz", "C3", "Cz", "C4"]
OUT = tempfile.gettempdir()


def written(extension: str) -> str:
    """The file a writer produced last; writers timestamp their names."""
    found = sorted(glob.glob(os.path.join(OUT, "s4e5_*" + extension)))
    return found[-1] if found else ""


if __name__ == "__main__":

    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE,
                          channel_count=len(ELECTRODES),
                          signal_frequency=10, signal_amplitude=50.0,
                          noise_amplitude=5.0)

    # Name the channels first, so there is metadata for the formats to
    # differ about; without this every format writes Ch01..Ch04
    labeler = gp.ChannelLabeler(labels=ELECTRODES)
    p.connect(source, labeler)

    # The same stream into four writers
    writers = {
        ".csv": gp.CsvWriter(file_name=os.path.join(OUT, "s4e5.csv")),
        ".edf": gp.EDFWriter(file_name=os.path.join(OUT, "s4e5.edf")),
        ".h5": gp.HDF5Writer(file_name=os.path.join(OUT, "s4e5.h5")),
        ".mat": gp.MatWriter(file_name=os.path.join(OUT, "s4e5.mat")),
    }
    for writer in writers.values():
        p.connect(labeler, writer)

    # What the writers were given, kept in memory to compare against
    recorded = gp.Collector(name="recorded")
    p.connect(labeler, recorded)

    p.start()
    time.sleep(SECONDS)
    p.stop()
    p.close()

    print(f"{SECONDS:.0f} s of {len(ELECTRODES)} channels at "
          f"{SAMPLING_RATE} Hz, written four ways:")
    for extension in writers:
        path = written(extension)
        size = os.path.getsize(path) if path else 0
        print(f"  {extension:<5} {size:>9,d} bytes  "
              f"{os.path.basename(path) or '(not written)'}")

    # Every format has a reader, and mode="batch" hands the whole file
    # back from one run(). HDF5Reader and MatReader take two arguments
    # the other two do not: variable_name names the array inside the
    # file, and has_time_row says its first row is a time axis
    readers = {
        ".csv": lambda path: gp.CsvReader(file_name=path, mode="batch"),
        ".edf": lambda path: gp.EDFReader(file_name=path, mode="batch"),
        ".h5": lambda path: gp.HDF5Reader(file_name=path, mode="batch",
                                          variable_name="data",
                                          has_time_row=True),
        ".mat": lambda path: gp.MatReader(file_name=path, mode="batch",
                                          variable_name="data",
                                          has_time_row=True),
    }
    reference = recorded.result.data
    print()
    print("Read back, against what the writers were given:")
    for extension, reader_for in readers.items():
        with gp.Pipeline() as back:
            back.connect(reader_for(written(extension)), gp.Collector())
            result = back.run()
        n = min(result.sample_count, len(reference))
        worst = np.max(np.abs(result.data[:n] - reference[:n]))
        print(f"  {extension:<5} largest difference {worst:<8.3g} uV  "
              f"labels {', '.join(result.labels)}")
    print()
    print("Files left in:", OUT)
