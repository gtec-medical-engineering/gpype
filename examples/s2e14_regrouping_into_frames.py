"""Season 2, Episode 14: a Framer regroups a stream into larger frames."""

import time

import gpype as gp

SAMPLING_RATE = 250
SOURCE_FRAME = 4  # what the source produces
WANTED_FRAME = 32  # what the node downstream wants


def cycles_by_name(load):
    """Cycle counts from a load report, keyed by node name."""
    return {row["name"]: row["cycles"] for row in load["nodes"]}


if __name__ == "__main__":

    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=4,
                          frame_size=SOURCE_FRAME, signal_amplitude=1.0)

    # Collects incoming samples and emits once it has WANTED_FRAME of them
    framer = gp.Framer(frame_size=WANTED_FRAME)
    before = gp.Collector(name="before_framer")
    after = gp.Collector(name="after_framer")

    p.connect(source, before)
    p.connect(source, framer)
    p.connect(framer, after)

    p.start()
    # The first reading includes the cycles run during setup: discard it
    p.get_load()
    time.sleep(2.0)
    load = p.get_load()
    p.stop()
    p.close()

    counts = cycles_by_name(load)
    print(f"source frame_size = {SOURCE_FRAME}, "
          f"Framer frame_size = {WANTED_FRAME}, "
          f"measured over {load['elapsed_s']:.2f} s")
    for name in ("Generator", "Collector 'before_framer'", "Framer",
                 "Collector 'after_framer'"):
        if name in counts:
            print(f"  {name:<28} {counts[name]:6d} cycles")

    # The collectors do not end level: up to WANTED_FRAME - 1 samples are
    # still inside the Framer when the pipeline stops
    for label, sink in (("before", before), ("after", after)):
        result = sink.result
        print(f"  {label:<8} {0 if result is None else len(result.data):6d}"
              " samples")
