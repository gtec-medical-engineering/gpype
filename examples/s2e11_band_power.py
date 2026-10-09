"""Season 2, Episode 11: BandPower reduces a spectrum to a few named bands."""

import time

import numpy as np

import gpype as gp

SAMPLING_RATE = 250
WINDOW = 250  # one second, so the bins are 1 Hz apart
SIGNAL_HZ = 10  # squarely inside alpha (8-13 Hz)


if __name__ == "__main__":

    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                          signal_frequency=SIGNAL_HZ, signal_amplitude=1.0,
                          noise_amplitude=0.1)

    spectrum = gp.FFT(window_size=WINDOW)
    # No argument: the five conventional EEG bands. Or bands=[[8, 12]]
    power = gp.BandPower()
    collected = gp.Collector(name="band_power")

    p.connect(source, spectrum)
    p.connect(spectrum, power)
    p.connect(power, collected)

    p.start()
    time.sleep(3.0)
    p.stop()
    p.close()

    result = collected.result
    print(f"a {SIGNAL_HZ} Hz sine: the power should sit in alpha")
    if result is None:
        print("nothing was collected")
    else:
        # The mean over the windows that arrived, so one noisy window
        # does not decide the answer
        block = result.data
        mean = np.asarray(block).reshape(len(block), -1).mean(axis=0)
        labels = result.labels or [f"col{i}" for i in range(len(mean))]
        biggest = int(np.argmax(mean))
        for i, (label, value) in enumerate(zip(labels, mean)):
            mark = "  <--" if i == biggest else ""
            print(f"  {label:<24} {value:10.4f}{mark}")
