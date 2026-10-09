"""Season 2, Episode 13: GenericFilter runs any coefficients scipy designs."""

import time

import numpy as np
from scipy.signal import butter, iirnotch

import gpype as gp

SAMPLING_RATE = 250
F_LO, F_HI = 9.0, 11.0
ORDER = 2  # Butterworth's default order in g.Pype


def collected(collector):
    """What a collector holds, as a flat array."""
    result = collector.result
    if result is None:
        return np.zeros(0)
    return np.asarray(result.data).reshape(-1)


if __name__ == "__main__":

    # Exactly what Bandpass computes: the cutoffs normalised by the
    # Nyquist frequency, half the sampling rate
    wn = [F_LO / SAMPLING_RATE * 2, F_HI / SAMPLING_RATE * 2]
    b, a = butter(N=ORDER, Wn=wn, btype="bandpass")

    # A 50 Hz notch with a quality factor, which no named filter offers
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

    print(f"Bandpass({F_LO:g}, {F_HI:g}) against butter(N={ORDER}, ...)")
    print(f"  samples compared     {n}")
    print(f"  largest difference   {difference:.3e}")
    print(f"  rms of the signal    {np.sqrt(np.mean(a_out ** 2)):.4f}")
    notch_rms = float(np.sqrt(np.mean(collected(notched) ** 2)))
    print(f"50 Hz notch, Q=30: output rms {notch_rms:.4f}")
