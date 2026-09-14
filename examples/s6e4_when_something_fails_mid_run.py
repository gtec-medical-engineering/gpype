# --------------------------------------------------------------
# Example file s6e4_when_something_fails_mid_run.py
# For details and usage, see g.Pype Training Season 6, Episode 4
# --------------------------------------------------------------
#
# Episode 2 showed a pipeline that could never run. The refusal reached you
# from start(), because it was knowable before a single sample flowed.
#
# This is the other kind of failure: the pipeline starts, runs for a
# while, and then a node raises on some frame. Nobody is waiting on a
# return value at that point -- the call came from a source thread -- so
# the failure cannot be raised at you. It is recorded, and it stops the
# run. This episode shows where to find it and what it left behind.
#
# The failing node is deliberately broken. Writing nodes properly is
# Season 7; this one exists only to break.

import glob
import os
import tempfile
import time

import gpype as gp

FILE = os.path.join(tempfile.gettempdir(), "s6e4_interrupted.csv")


class Flaky(gp.IONode):
    """Passes data through, then raises on the twentieth frame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._seen = 0

    def step(self, data):
        self._seen += 1
        if self._seen == 20:
            raise RuntimeError("the twentieth frame was too much for me")
        return {gp.Constants.Defaults.PORT_OUT:
                data[gp.Constants.Defaults.PORT_IN]}


if __name__ == "__main__":

    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=250, channel_count=4)
    flaky = Flaky(name="flaky")
    p.connect(source, flaky)
    p.connect(flaky, gp.CsvWriter(file_name=FILE))

    # start() returns normally. The pipeline was viable and the run really
    # did start -- the failure has not happened yet.
    p.start()
    print("started; the pipeline was fine")

    # Wait for it to go wrong. In a real application this is the same
    # check a status indicator would poll.
    deadline = time.time() + 10
    while time.time() < deadline:
        if p.get_condition() == gp.Constants.Conditions.ERROR:
            break
        time.sleep(0.05)

    print()
    print("condition :", p.get_condition())
    failure = p.failure
    if failure:
        # A log entry, not a dict, but you subscript it the same way.
        print("message   :", failure["message"])
        print("node      :", failure["source"].get("instance"))
        print("raised in :", failure["source"].get("summary"))

    p.stop()
    p.close()

    # CsvWriter appends a timestamp to the name it was given, so the
    # file on disk is s6e4_interrupted_<date>_<time>.csv. Two header
    # lines precede the samples.
    written = sorted(glob.glob(FILE.replace(".csv", "_*.csv")))
    rows = 0
    if written:
        with open(written[-1]) as f:
            rows = sum(1 for _ in f) - 2

    print()
    print("Three things to take away.")
    print()
    print("One: a failing node stops the whole pipeline. The first error")
    print("ends the run -- g.Pype does not carry on with the rest of the")
    print("pipeline, because downstream nodes would go on emitting results")
    print("computed from data that stopped arriving.")
    print()
    print("Two: the recording is short, not corrupt. The stop closes the")
    print("file properly, so")
    print(f"  {written[-1] if written else FILE}")
    print(f"holds {rows} valid rows -- every sample up to the frame that")
    print("failed, and nothing after it -- and simply ends early. Check")
    print("the condition before you trust a run that finished quietly; a")
    print("short file is the only other clue you get.")
    print()
    print("Three: the entry names the node. That is what the name=")
    print("argument buys you. 'flaky' is findable; a pipeline with three")
    print("unnamed nodes of the same class is not.")
