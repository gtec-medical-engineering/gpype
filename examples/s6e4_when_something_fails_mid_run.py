"""Season 6, Episode 4: a node that fails mid-run, and what it leaves behind.

The failing node is deliberately broken and exists only to break;
writing nodes properly is Season 7.
"""

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

    # start() returns normally: the pipeline was viable and the run did
    # start; the failure has not happened yet
    p.start()
    print("started; the pipeline was fine")

    # Wait for it to go wrong, the same check a status indicator polls
    deadline = time.time() + 10
    while time.time() < deadline:
        if p.get_condition() == gp.Constants.Conditions.ERROR:
            break
        time.sleep(0.05)

    print()
    print("condition :", p.get_condition())
    failure = p.failure
    if failure:
        # A log entry, subscripted like a dict
        print("message   :", failure["message"])
        print("node      :", failure["source"].get("instance"))
        print("raised in :", failure["source"].get("summary"))

    p.stop()
    p.close()

    # The writer appended a timestamp to the name, and closed the file
    # properly: comment lines, one header line, then the rows
    written = sorted(glob.glob(FILE.replace(".csv", "_*.csv")))
    rows = 0
    if written:
        with open(written[-1]) as handle:
            rows = sum(1 for line in handle if not line.startswith("#")) - 1
    print()
    print("file      :", os.path.basename(written[-1]) if written else "-")
    print("rows      :", rows)
