# --------------------------------------------------------------
# Example file s6e5_when_the_numbers_are_wrong.py
# For details and usage, see g.Pype Training Season 6, Episode 5
# --------------------------------------------------------------
#
# The earlier episodes were about a pipeline that said something was
# wrong. This one is the harder case: nothing is wrong, and the answer
# is still not the one you wanted.
#
# A filter pointed at the wrong band is not a defect. It does exactly
# what it was told, raises nothing, and leaves the pipeline Healthy. No
# message can help you here, because there is no message to give. The
# only thing that knows is the data.
#
# So the technique is to tap the signal at more than one point and
# compare one number, instead of staring at a scope and hoping.

import time

import numpy as np

import gpype as gp

SAMPLING_RATE = 250
SIGNAL_HZ = 10


def rms(collector) -> float:
    """Root mean square of the second half of what was collected.

    The second half only: every filter needs a moment to settle, and
    the transient at the start is not the steady state you are trying
    to measure.
    """
    result = collector.result
    if result is None:
        return float("nan")
    block = result.data
    return float(np.sqrt(np.mean(np.square(block[len(block) // 2:]))))


if __name__ == "__main__":

    p = gp.Pipeline()
    # signal_amplitude defaults to 0.0, so a Generator asked for
    # nothing else emits silence -- worth knowing before you conclude
    # that a filter ate your signal.
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=4,
                          signal_frequency=SIGNAL_HZ,
                          signal_amplitude=1.0)

    # A Collector terminates a pipeline in memory rather than in a file,
    # which is what makes a tap cheap enough to add three of.
    raw = gp.Collector(name="raw")
    in_band = gp.Collector(name="in_band")
    out_of_band = gp.Collector(name="out_of_band")

    # One source, three taps: the signal as it arrives, and the output
    # of two filters. Only one of the two is aimed at the signal.
    good = gp.Bandpass(f_lo=9, f_hi=11, name="alpha")
    bad = gp.Bandpass(f_lo=20, f_hi=30, name="beta")

    p.connect(source, raw)
    p.connect(source, good)
    p.connect(good, in_band)
    p.connect(source, bad)
    p.connect(bad, out_of_band)

    p.start()
    time.sleep(2.0)
    p.stop()
    p.close()

    r_raw = rms(raw)
    r_good = rms(in_band)
    r_bad = rms(out_of_band)

    print()
    print(f"signal is a {SIGNAL_HZ} Hz sine at {SAMPLING_RATE} Hz")
    print()
    print(f"  raw                      rms = {r_raw:8.4f}")
    print(f"  Bandpass  9-11 Hz        rms = {r_good:8.4f}"
          f"   ({100 * r_good / r_raw:5.1f} % of raw)")
    print(f"  Bandpass 20-30 Hz        rms = {r_bad:8.4f}"
          f"   ({100 * r_bad / r_raw:5.1f} % of raw)")
    print()
    print("condition:", p.get_condition())
    print()
    print("The pipeline is Healthy. Both filters worked. One of them")
    print("simply had nothing to pass, because the band it was given")
    print("does not contain the signal -- and that is a decision, not")
    print("a fault, so nothing raised and nothing was logged.")
    print()
    print("What found it was the comparison. A single tap tells you a")
    print("number; two tell you whether the number is plausible. When")
    print("a result looks wrong, add a Collector either side of the")
    print("node you suspect and compare one scalar. That narrows a")
    print("whole pipeline to one node far faster than reading code.")
    print()
    print("Two habits worth keeping:")
    print()
    print("Skip the transient. A filter's first samples are its")
    print("settling, not its answer, which is why rms() above uses")
    print("only the second half of the block.")
    print()
    print("Name your taps. The Collectors here are 'raw', 'in_band'")
    print("and 'out_of_band', so the printout reads as a measurement")
    print("rather than as three anonymous arrays.")
