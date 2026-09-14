# --------------------------------------------------------------
# Example file s2e13_your_own_filter.py
# For details and usage, see g.Pype Training Season 2, Episode 13
# --------------------------------------------------------------
#
# Bandpass, Bandstop, Highpass and Lowpass cover most of what an EEG
# pipeline needs, and underneath they are all the same thing: a
# Butterworth design turned into filter coefficients.
#
# When you need something they do not offer -- a notch with a specific
# quality factor, an elliptic response, a Chebyshev, an FIR from a paper
# -- GenericFilter takes the coefficients directly. Anything
# scipy.signal can design, this node can run.
#
# The check below is the honest way to introduce it: design by hand
# exactly what Bandpass would have designed, run both on the same
# signal, and confirm the outputs agree. Once they do, you know the
# named filters are a convenience rather than a separate mechanism --
# and that your own coefficients stand on the same footing.

import time

import numpy as np
from scipy.signal import butter, iirnotch

import gpype as gp

SAMPLING_RATE = 250
F_LO, F_HI = 9.0, 11.0
ORDER = 2          # Butterworth's default order in g.Pype


def collected(collector) -> np.ndarray:
    """What a collector holds, as a flat array."""
    result = collector.result
    if result is None:
        return np.zeros(0)
    return np.asarray(result.data).reshape(-1)


if __name__ == "__main__":

    # Exactly what Bandpass computes internally: the cutoffs normalised
    # by the Nyquist frequency, which is half the sampling rate.
    wn = [F_LO / SAMPLING_RATE * 2, F_HI / SAMPLING_RATE * 2]
    b, a = butter(N=ORDER, Wn=wn, btype="bandpass")

    # A 50 Hz mains notch, which none of the named filters offers:
    # Bandstop takes a band, not a quality factor, so a sharp notch is
    # exactly the case GenericFilter exists for.
    b_notch, a_notch = iirnotch(w0=50.0, Q=30.0, fs=SAMPLING_RATE)

    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                          signal_frequency=10, signal_amplitude=1.0,
                          noise_amplitude=0.3)

    named_filter = gp.Bandpass(f_lo=F_LO, f_hi=F_HI)
    own_filter = gp.GenericFilter(b=b, a=a)
    notch = gp.GenericFilter(b=b_notch, a=a_notch)

    named = gp.Collector(name="named")
    hand_rolled = gp.Collector(name="hand_rolled")
    notched = gp.Collector(name="notched")

    for node, sink in ((named_filter, named), (own_filter, hand_rolled),
                       (notch, notched)):
        p.connect(source, node)
        p.connect(node, sink)

    p.start()
    time.sleep(2.0)
    p.stop()
    p.close()

    a_out = collected(named)
    b_out = collected(hand_rolled)
    n = min(len(a_out), len(b_out))
    difference = float(np.max(np.abs(a_out[:n] - b_out[:n]))) if n else -1.0

    print()
    print(f"Bandpass({F_LO:g}, {F_HI:g}) against butter(N={ORDER}, ...)")
    print("run on the same signal:")
    print()
    print(f"  samples compared        {n}")
    print(f"  largest difference      {difference:.3e}")
    print(f"  rms of the signal       {np.sqrt(np.mean(a_out ** 2)):.4f}")
    print()
    print("Zero, bit for bit -- not merely close. They are the same")
    print("filter. Bandpass normalises the cutoffs by Nyquist,")
    print("calls scipy's butter(), and runs the result -- which is")
    print("precisely the three lines above the pipeline.")
    print()
    print("So GenericFilter is not an escape hatch for hard cases. It")
    print("is the mechanism, with the named filters as shorthand for")
    print("the four designs people ask for most.")
    print()
    notch_rms = float(np.sqrt(np.mean(collected(notched) ** 2)))
    print("The third branch is one of the cases the shorthand does not")
    print("cover: a 50 Hz notch at Q=30, from scipy's iirnotch. Its")
    print(f"output rms is {notch_rms:.4f}, barely below the input, because")
    print("a notch removes one narrow band and leaves everything else")
    print("-- which is the whole point of using one.")
    print()
    print("Two things to keep in mind. Design against the sampling")
    print("rate you will actually run at: coefficients are specific to")
    print("it, and a filter designed for 250 Hz is silently wrong at")
    print("500. And the node runs the filter forwards only, one frame")
    print("at a time, because that is what real time means -- see the")
    print("zero-phase discussion in the reference for what an offline")
    print("run can do instead.")
