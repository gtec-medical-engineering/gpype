# --------------------------------------------------------------
# Example file s2e12_crossing_a_threshold.py
# For details and usage, see g.Pype Training Season 2, Episode 12
# --------------------------------------------------------------
#
# At some point a measurement has to become a decision: is the feature
# high enough, yes or no. That step is where a well-behaved pipeline
# usually starts misbehaving, because a bare comparison flips on every
# sample that grazes the level -- and a noisy signal grazes it a lot.
#
# Threshold does the comparison, and its other two parameters exist
# solely to stop the chattering. This episode counts the transitions so
# the difference is a number rather than an opinion.

import time

import numpy as np

import gpype as gp

SAMPLING_RATE = 250
SECONDS = 4.0
SLOW_HZ = 0.5     # 2 full cycles in 4 s, so 4 honest crossings of zero


def transitions(collector) -> int:
    """How many times the held state changed."""
    result = collector.result
    if result is None:
        return -1
    state = np.asarray(result.data).reshape(-1)
    return int(np.count_nonzero(np.diff(state)))


if __name__ == "__main__":

    p = gp.Pipeline()

    # A slow swing that crosses zero four times in four seconds, buried
    # in noise big enough to cross it many more times by accident.
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                          signal_frequency=SLOW_HZ, signal_amplitude=1.0,
                          noise_amplitude=0.4)

    bare = gp.Collector(name="bare")
    with_hysteresis = gp.Collector(name="hysteresis")
    with_dwell = gp.Collector(name="dwell")
    wide = gp.Collector(name="wide_hysteresis")

    # Three decisions about the same signal.
    plain = gp.Threshold(level=0.0)
    # Leaving the state needs the signal to fall well back below it, so
    # noise that entered the state cannot immediately leave it again.
    hyst = gp.Threshold(level=0.2, release=-0.2)
    # The new state must hold for 25 samples -- a tenth of a second --
    # before it is believed at all.
    dwell = gp.Threshold(level=0.0, dwell=25)
    # The same idea as hyst, sized against the noise instead of below
    # it: the gap is two standard deviations either way.
    hyst_wide = gp.Threshold(level=0.8, release=-0.8)

    for node, sink in ((plain, bare), (hyst, with_hysteresis),
                       (hyst_wide, wide), (dwell, with_dwell)):
        p.connect(source, node)
        p.connect(node, sink)

    p.start()
    time.sleep(SECONDS)
    p.stop()
    p.close()

    print()
    print(f"a {SLOW_HZ} Hz swing over {SECONDS:.0f} s -- four honest")
    print("crossings of zero -- with noise of standard deviation 0.4")
    print()
    print(f"  Threshold(level=0)                    "
          f"{transitions(bare):5d} transitions")
    print(f"  Threshold(level=0.2, release=-0.2)    "
          f"{transitions(with_hysteresis):5d} transitions")
    print(f"  Threshold(level=0.8, release=-0.8)    "
          f"{transitions(wide):5d} transitions")
    print(f"  Threshold(level=0, dwell=25)          "
          f"{transitions(with_dwell):5d} transitions")
    print()
    print("Four is the answer the signal deserves. The bare comparison")
    print("is not wrong about any single sample -- every one of those")
    print("flips really did cross zero -- it is simply answering a")
    print("question nobody asked.")
    print()
    print("Hysteresis gives leaving the state its own, lower level, so")
    print("getting out takes more than the noise that got you in. Note")
    print("that it only helps if the gap is sized against the noise:")
    print("plus or minus 0.2 against a standard deviation of 0.4 is")
    print("inside the noise, and barely helps. Widening it to plus or")
    print("minus 0.8 -- two standard deviations either way -- is what")
    print("makes it work. Hysteresis chosen by eye is a knob that")
    print("looks tuned and is not.")
    print()
    print("Dwell requires the new state to hold before it is believed.")
    print("Use it when brief excursions are real but uninteresting. It")
    print("costs latency by construction: 25 samples at 250 Hz is 100")
    print("ms of delay on every decision, which for a control signal a")
    print("user is watching is a decision you should make on purpose.")
    print()
    print("The output is 0.0 or 1.0 in the pipeline's own data type,")
    print("not a boolean array, so it flows on into the rest of the")
    print("pipeline like any other channel.")
