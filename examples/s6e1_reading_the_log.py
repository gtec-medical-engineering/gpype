# --------------------------------------------------------------
# Example file s6e1_reading_the_log.py
# For details and usage, see g.Pype Training Season 6, Episode 1
# --------------------------------------------------------------
#
# A running pipeline cannot raise at you. Its nodes are driven from
# source threads that nobody is waiting on, so when one of them has
# something to say -- a warning, a refusal, an exception -- it is
# written to the session log instead. That file is the only channel a
# realtime run has, and a windowed application may have no console at
# all.
#
# So this is the first episode of the season for a reason: knowing where
# the log is, and how to read a line of it, is what makes the rest of
# the season usable.

import glob
import os
import sys
import time

import gpype as gp


def log_directory() -> str:
    """Where g.Pype writes its session log on this platform."""
    if sys.platform == "win32":
        return os.path.join(os.getenv("APPDATA", ""), "gtec", "gPype")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/gtec/gPype")
    return os.path.expanduser("~/.local/share/gtec/gPype")


def newest_log(directory: str) -> str:
    """The log file this run most likely wrote."""
    found = sorted(glob.glob(os.path.join(directory, "*.log")))
    return found[-1] if found else ""


if __name__ == "__main__":

    p = gp.Pipeline()

    # Two filters of the same class. Name them: without a name both are
    # "Bandpass" in the log and you cannot tell which one spoke.
    source = gp.Generator(sampling_rate=250, channel_count=4,
                          signal_amplitude=1.0)
    alpha = gp.Bandpass(f_lo=8, f_hi=12, name="alpha")
    beta = gp.Bandpass(f_lo=13, f_hi=30, name="beta")

    p.connect(source, alpha)
    p.connect(alpha, gp.Collector())
    p.connect(source, beta)
    p.connect(beta, gp.Collector())

    p.start()
    time.sleep(1.0)
    p.stop()
    p.close()

    directory = log_directory()
    path = newest_log(directory)

    print()
    print("Log directory:", directory)
    print("Newest file  :", os.path.basename(path) or "(none found)")
    print()
    print("A line has five columns:")
    print()
    print("  timestamp | type | source | thread | message")
    print()

    if path:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = [line.rstrip("\n") for line in handle if line.strip()]
        print("The last few lines this run wrote:")
        print()
        for line in lines[-8:]:
            print("  " + line[:150])
        print()

    print("Two columns earn their place.")
    print()
    print("source names the node *instance* where you gave one, so")
    print("'alpha' and 'beta' above are told apart. Without a name it")
    print("falls back to the class, and three Bandpass nodes in one")
    print("pipeline all look identical -- which is the whole reason to")
    print("pass name= to anything you might later have to ask about.")
    print()
    print("thread is worth reading when the ordering surprises you. A")
    print("pipeline runs its sources on their own threads, so lines")
    print("from different parts of the pipeline interleave, and two")
    print("entries next to each other are not necessarily related.")
    print()
    print("One thing this file is not is a place to look afterwards and")
    print("hope. A failure stops the run -- see Episode 4 -- so if you")
    print("want to know whether a run was clean, ask the pipeline")
    print("directly with get_condition() rather than reading the log")
    print("later. The log tells you *what* went wrong; the condition")
    print("tells you *that* something did.")
