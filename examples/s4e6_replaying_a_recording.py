# --------------------------------------------------------------
# Example file s4e6_replaying_a_recording.py
# For details and usage, see g.Pype Training Season 4, Episode 6
# --------------------------------------------------------------
#
# Every format g.Pype writes, it can also read. That turns a recording
# into a source, which is worth more than it first sounds:
#
#   - you can develop a pipeline without the amplifier plugged in, and
#     without a subject sitting in the chair;
#   - you can replay the same data after a change and see whether the
#     answer moved, which is impossible with live signal;
#   - and you can process a whole recording offline, as fast as the
#     machine allows, instead of in real time.
#
# The last one is a different execution mode, not merely a faster
# setting, and this episode shows both.

import glob
import os
import tempfile
import time

import gpype as gp

SAMPLING_RATE = 250
SECONDS = 4.0
OUT = tempfile.gettempdir()
STEM = os.path.join(OUT, "s4e6_recording.csv")


def newest(pattern: str) -> str:
    found = sorted(glob.glob(pattern))
    return found[-1] if found else ""


def record() -> str:
    """Make a recording to read back, and return its path."""
    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=4,
                          signal_frequency=10, signal_amplitude=50.0,
                          noise_amplitude=5.0)
    # The writer hangs off the *labeler*, not off the source: names
    # only reach a file if the file is downstream of the node that
    # assigns them.
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
    print()
    print("recorded:", os.path.basename(path))

    # ---- 1. Replay in real time -------------------------------------
    #
    # The default. The reader paces the samples out as if they were
    # arriving now, which is what you want when the thing you are
    # testing cares about timing -- a scope, a paradigm, anything with
    # a deadline. speed= multiplies that pace; loop= starts over at the
    # end, so a short recording can drive a long test.
    started = time.time()
    p = gp.Pipeline()
    reader = gp.CsvReader(file_name=path, speed=2.0)
    live = gp.Collector(name="replayed")
    p.connect(reader, live)
    p.start()
    time.sleep(SECONDS / 4)      # a quarter of the wall time it took
    p.stop()
    p.close()
    realtime_seconds = time.time() - started
    replayed = live.result

    # ---- 2. Process the whole thing offline -------------------------
    #
    # mode='batch' makes the reader a batch source, and the pipeline is
    # then driven by run() rather than started: it returns once the last
    # sample has passed the last node. No threads, no pacing, no waiting
    # -- and an exception in a node arrives as an ordinary traceback
    # through this line, because the run is on this thread.
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
    print()
    print("The realtime replay stopped part-way because the script")
    print("stopped it: at speed=2.0 it was working through the file at")
    print("twice life speed and had got as far as it had got. The batch")
    print("run stopped because the recording ran out. Nobody had to")
    print("guess how long to wait, which is the first reason to prefer")
    print("it for analysis -- and it took a fraction of the time,")
    print("because there was no pacing to honour.")
    print()
    print("run() also hands the data back. With one Collector in the")
    print("pipeline it returns that Result; with several it returns a dict")
    print("keyed by node name; with none it returns None and the output")
    print("went wherever the sinks put it.")
    if result is not None:
        print()
        print("  channels :", result.channel_count)
        print("  labels   :", result.labels)
        print("  rate     :", result.rate)
    print()
    print("Notice the labels survived the round trip. They were written")
    print("into the CSV header by the recording pipeline and read back")
    print("out by the reader, so an analysis can still ask for Cz by")
    print("name -- which is the whole argument for naming channels at")
    print("the source rather than in the analysis script.")
    print()
    print("Every reader takes the same arguments -- file_name,")
    print("sampling_rate, frame_size, speed, loop and mode -- so what")
    print("you learn here applies to EDFReader, HDF5Reader and")
    print("MatReader unchanged.")
