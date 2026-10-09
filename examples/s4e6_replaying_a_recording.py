"""Season 4, Episode 6: a recording replayed live, then run offline."""

import glob
import os
import tempfile
import time

import gpype as gp

SAMPLING_RATE = 250
SECONDS = 4.0
STEM = os.path.join(tempfile.gettempdir(), "s4e6.csv")


def newest(pattern: str) -> str:
    """The file matching the pattern that was written last."""
    found = sorted(glob.glob(pattern))
    return found[-1] if found else ""


def record() -> str:
    """Make a recording to read back, and return its path."""
    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=4,
                          signal_frequency=10, signal_amplitude=50.0,
                          noise_amplitude=5.0)
    # The writer hangs off the labeler: names reach a file only when
    # the file is downstream of the node that assigns them
    labeler = gp.ChannelLabeler(labels=["Fz", "C3", "Cz", "C4"])
    p.connect(source, labeler)
    p.connect(labeler, gp.CsvWriter(file_name=STEM))
    p.start()
    time.sleep(SECONDS)
    p.stop()
    p.close()
    return newest(STEM.replace(".csv", "_*.csv"))


if __name__ == "__main__":

    path = record()
    print("recorded:", os.path.basename(path))

    # 1. Replay in real time, the default: the reader paces the samples
    # out as if they were arriving now. speed multiplies that pace and
    # loop starts over at the end
    started = time.time()
    p = gp.Pipeline()
    reader = gp.CsvReader(file_name=path, speed=2.0)
    live = gp.Collector(name="replayed")
    p.connect(reader, live)
    p.start()
    time.sleep(SECONDS / 4)  # stopped by the script, not by the file
    p.stop()
    p.close()
    realtime_seconds = time.time() - started
    replayed = live.result

    # 2. Process the whole thing offline: mode="batch" makes the reader
    # a batch source, and run() drives the pipeline to the end on this
    # thread and returns what the Collector gathered
    started = time.time()
    p = gp.Pipeline()
    offline_reader = gp.CsvReader(file_name=path, mode="batch")
    p.connect(offline_reader, gp.Collector(name="offline"))
    result = p.run()
    p.close()
    batch_seconds = time.time() - started

    print()
    print("  realtime replay at speed=2.0")
    print(f"      {len(replayed.data) if replayed else 0:6d} samples in "
          f"{realtime_seconds:.2f} s of wall time")
    print("  batch run")
    print(f"      {len(result.data) if result is not None else 0:6d} "
          f"samples in {batch_seconds:.2f} s of wall time")
    if result is not None:
        print()
        print("  channels :", result.channel_count)
        print("  labels   :", result.labels)
        print("  rate     :", result.rate)
