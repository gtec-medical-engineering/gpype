"""Season 6, Episode 3: how close a pipeline is to its deadline."""

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
    # The verdict, not just the number: a node listed here spends more
    # than LOAD_WARNING_DUTY of the budget on its own step()
    short = load["short_of_headroom"]
    print("  short of headroom : %s" % (", ".join(short) if short else "-"))
    for row in sorted(load["nodes"], key=lambda r: -(r["duty"] or 0)):
        print("    %-24s duty=%.4f  worst=%.2f ms"
              % (row["name"], row["duty"], row["worst_cycle_ms"]))


if __name__ == "__main__":

    # A modest pipeline: one source, one filter, one sink
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

    # A much heavier one. How far the headroom falls depends on the
    # machine, so read which node is busiest and the worst single cycle
    # rather than the absolute number
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
