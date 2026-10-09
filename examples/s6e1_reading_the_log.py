"""Season 6, Episode 1: where the session log is, and how to read a line."""

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

    # Two filters of the same class, named: without a name both are
    # "Bandpass" in the log and you cannot tell which one spoke
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

    print("Log directory:", directory)
    print("Newest file  :", os.path.basename(path) or "(none found)")
    # A line has five columns: timestamp | type | source | thread | message
    if path:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = [line.rstrip("\n") for line in handle if line.strip()]
        print()
        print("The last few lines this run wrote:")
        for line in lines[-8:]:
            print("  " + line[:150])
