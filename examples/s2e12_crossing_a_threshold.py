"""Season 2, Episode 12: a Threshold turns a feature into a decision."""

import time

import numpy as np

import gpype as gp

SAMPLING_RATE = 250
SECONDS = 4.0
SLOW_HZ = 0.5  # 2 full cycles in 4 s, so 4 honest crossings of zero


def transitions(collector):
    """How many times the held state changed."""
    result = collector.result
    if result is None:
        return -1
    state = np.asarray(result.data).reshape(-1)
    return int(np.count_nonzero(np.diff(state)))


if __name__ == "__main__":

    p = gp.Pipeline()

    # A slow swing that crosses zero four times in four seconds, in noise
    # big enough to cross it many more times by accident
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                          signal_frequency=SLOW_HZ, signal_amplitude=1.0,
                          noise_amplitude=0.4)

    # Four decisions about the same signal
    plain = gp.Threshold(level=0.0)
    # Hysteresis: leaving the state needs the signal well back below it.
    # A gap inside the noise barely helps; two standard deviations do
    narrow = gp.Threshold(level=0.2, release=-0.2)
    wide = gp.Threshold(level=0.8, release=-0.8)
    # Dwell: the new state must hold for 25 samples before it is believed
    dwell = gp.Threshold(level=0.0, dwell=25)

    kept = {}
    for name, node in (("plain", plain), ("narrow", narrow),
                       ("wide", wide), ("dwell", dwell)):
        kept[name] = gp.Collector(name=name)
        p.connect(source, node)
        p.connect(node, kept[name])

    p.start()
    time.sleep(SECONDS)
    p.stop()
    p.close()

    print(f"a {SLOW_HZ} Hz swing over {SECONDS:.0f} s, four crossings of "
          "zero, noise of standard deviation 0.4")
    print(f"  Threshold(level=0)                  "
          f"{transitions(kept['plain']):5d} transitions")
    print(f"  Threshold(level=0.2, release=-0.2)  "
          f"{transitions(kept['narrow']):5d} transitions")
    print(f"  Threshold(level=0.8, release=-0.8)  "
          f"{transitions(kept['wide']):5d} transitions")
    print(f"  Threshold(level=0, dwell=25)        "
          f"{transitions(kept['dwell']):5d} transitions")
