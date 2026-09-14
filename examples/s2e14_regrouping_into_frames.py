# --------------------------------------------------------------
# Example file s2e14_regrouping_into_frames.py
# For details and usage, see g.Pype Training Season 2, Episode 14
# --------------------------------------------------------------
#
# A source decides how many samples travel together. That number, the
# frame size, is chosen for acquisition: small enough to keep latency
# low, large enough that the pipeline is not woken thousands of times a
# second.
#
# Some algorithms want a different number. An FFT wants its window; an
# epoch wants its epoch. Framer regroups the stream so a node downstream
# receives the frame size it needs, instead of every such node keeping a
# buffer of its own and getting the bookkeeping subtly wrong.
#
# Framer can only aggregate. It collects incoming samples and emits when
# it has enough, so the output frame is a multiple of what arrives; it
# cannot hand out fewer samples than it was given.

import time

import gpype as gp

SAMPLING_RATE = 250
SOURCE_FRAME = 4       # what the source produces
WANTED_FRAME = 32      # what the node downstream wants


def cycles_by_name(load: dict) -> dict:
    """Cycle counts from a load report, keyed by node name."""
    return {row["name"]: row["cycles"] for row in load["nodes"]}


if __name__ == "__main__":

    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=4,
                          frame_size=SOURCE_FRAME, signal_amplitude=1.0)

    framer = gp.Framer(frame_size=WANTED_FRAME)
    before = gp.Collector(name="before_framer")
    after = gp.Collector(name="after_framer")

    p.connect(source, before)
    p.connect(source, framer)
    p.connect(framer, after)

    p.start()
    # Discard the first reading: it covers everything since the nodes
    # were built, setup included, and is not representative.
    p.get_load()
    time.sleep(2.0)
    load = p.get_load()
    p.stop()
    p.close()

    counts = cycles_by_name(load)
    seconds = load["elapsed_s"]

    print()
    print(f"source frame_size = {SOURCE_FRAME}, "
          f"Framer frame_size = {WANTED_FRAME}")
    print(f"measured over {seconds:.2f} s")
    print()
    for name in ("Generator", "Collector 'before_framer'", "Framer",
                 "Collector 'after_framer'"):
        if name in counts:
            print(f"  {name:<28} {counts[name]:6d} cycles")
    print()
    print("The counts tell the story. The Framer is driven as often as")
    print("everything else -- it sees every frame the source produces")
    print("-- but the collector behind it wakes eight times less often,")
    print(f"because {WANTED_FRAME} // {SOURCE_FRAME} = "
          f"{WANTED_FRAME // SOURCE_FRAME} input frames go into one")
    print("output frame.")
    print()
    print("No samples are lost or invented, but the two collectors do")
    print("not end level:")
    print()
    counted = {}
    for label, sink in (("before", before), ("after", after)):
        result = sink.result
        counted[label] = 0 if result is None else len(result.data)
        print(f"  {label:<8} {counted[label]:6d} samples")
    print()
    print(f"The difference is {counted['before'] - counted['after']}"
          " samples, and it is where you would")
    print("expect it: still inside the Framer. A frame is emitted only")
    print(f"once all {WANTED_FRAME} samples are there, so at any instant the")
    print(f"Framer is holding up to {WANTED_FRAME - 1} samples that have")
    print("arrived and not yet left. Stopping does not flush them.")
    print()
    print("That matters when a run ends: the last partial frame is not")
    print("in the recording downstream of a Framer. Whether that is")
    print("acceptable is a question about your experiment, not about")
    print("the node -- but it is a question, and it has to be asked.")
    print()
    print("Two consequences worth stating out loud.")
    print()
    print("Latency. A frame of 32 at 250 Hz cannot exist before its")
    print("last sample does, so anything after the Framer sees data at")
    print("least 128 ms old. That is not overhead, it is arithmetic --")
    print("but it is arithmetic you should do before promising a")
    print("response time.")
    print()
    print("Direction. Framer aggregates and only aggregates. If a node")
    print("needs *smaller* frames than the source produces, the answer")
    print("is a smaller frame size at the source, not a Framer.")
