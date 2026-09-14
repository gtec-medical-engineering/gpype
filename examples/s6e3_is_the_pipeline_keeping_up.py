# --------------------------------------------------------------
# Example file s6e3_is_the_pipeline_keeping_up.py
# For details and usage, see g.Pype Training Season 6, Episode 3
# --------------------------------------------------------------
#
# A realtime pipeline has a deadline. Every frame has to be processed
# before the next one arrives, and if it is not, the backlog grows until
# the pipeline is stopped for you.
#
# Pipeline.get_load() tells you how close to that deadline you are,
# before it becomes a problem. Headroom of 1.0 means idle; 0.0 means
# there is nothing left.

import time

import gpype as gp


def report(pipeline, title):
    """Print one load reading."""
    load = pipeline.get_load()
    print()
    print("---", title)
    print("  headroom : %.3f   (1.0 = idle, 0.0 = saturated)"
          % load["headroom"])
    print("  duty     : %.3f" % load["duty"])
    print("  busiest  : %s" % load["busiest"])
    # The verdict, not just the number: a node listed here is spending
    # more than LOAD_WARNING_DUTY of the budget on its own step().
    short = load["short_of_headroom"]
    print("  short of headroom : %s" % (", ".join(short) if short else "-"))
    for row in sorted(load["nodes"], key=lambda r: -(r["duty"] or 0)):
        print("    %-24s duty=%.4f  worst=%.2f ms"
              % (row["name"], row["duty"], row["worst_cycle_ms"]))


if __name__ == "__main__":

    # A modest pipeline: one source, one filter, one sink.
    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=250, channel_count=8)
    band = gp.Bandpass(f_lo=8, f_hi=12)
    p.connect(source, band)
    p.connect(band, gp.Collector())
    p.start()
    time.sleep(1.0)
    report(p, "one filter")
    p.stop()
    p.close()

    # Now a much heavier pipeline. How far the headroom falls depends on
    # your machine, so do not read the absolute number as a verdict --
    # read which node is busiest, and the worst single cycle.
    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=1000, channel_count=256)
    stage = source
    for _ in range(12):
        nxt = gp.Bandpass(f_lo=1, f_hi=40)
        p.connect(stage, nxt)
        stage = nxt
    p.connect(stage, gp.Collector())
    p.start()
    time.sleep(1.0)
    report(p, "256 channels at 1 kHz through twelve filters")
    p.stop()
    p.close()

    print()
    print("Two things to take from this.")
    print()
    print("Each number is that node's own time, with everything it")
    print("feeds excluded, so the busiest node is where the work is --")
    print("not merely the one furthest downstream.")
    print()
    print("The source usually leads, and that is not a defect. A source")
    print("paced by the clock wakes once per *sample*, while everything")
    print("downstream wakes once per *frame*. At 1000 Hz with frames of")
    print("16 that is 1000 cycles a second against 62, so the source")
    print("pays sixteen times the per-cycle overhead. Compare a node")
    print("against its own history, not against the source.")
    print()
    print("For a deadline, worst_cycle_ms is the number that matters:")
    print("one frame at 250 Hz with frames of 4 is 16 ms, so a worst")
    print("cycle near that is late even if the average looks fine.")
