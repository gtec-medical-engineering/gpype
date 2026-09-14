# --------------------------------------------------------------
# Example file s2e11_band_power.py
# For details and usage, see g.Pype Training Season 2, Episode 11
# --------------------------------------------------------------
#
# A spectrum has one number per frequency bin, which is far more than
# any decision needs. Almost every online BCI reduces it to a handful of
# named bands first -- how much alpha, how much beta -- and works with
# those.
#
# BandPower does that reduction. It sits after an FFT, takes the bands
# by name (the conventional EEG bands are built in) or as explicit
# ranges, and emits one value per band per channel, labelled, so that a
# recording says which column is which instead of leaving you to count.

import time

import numpy as np

import gpype as gp

SAMPLING_RATE = 250
WINDOW = 250          # one second, so the bins are 1 Hz apart
SIGNAL_HZ = 10        # squarely inside alpha (8-13 Hz)


if __name__ == "__main__":

    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                          signal_frequency=SIGNAL_HZ, signal_amplitude=1.0,
                          noise_amplitude=0.1)

    spectrum = gp.FFT(window_size=WINDOW)
    power = gp.BandPower()          # the five conventional EEG bands
    collected = gp.Collector(name="band_power")

    p.connect(source, spectrum)
    p.connect(spectrum, power)
    p.connect(power, collected)

    p.start()
    time.sleep(3.0)
    p.stop()
    p.close()

    result = collected.result
    print()
    print(f"a {SIGNAL_HZ} Hz sine, so the power should sit in alpha")
    print()
    if result is None:
        print("nothing was collected")
    else:
        block = result.data
        # Average over the windows that arrived, so one noisy window
        # does not decide the answer.
        mean = np.asarray(block).reshape(len(block), -1).mean(axis=0)
        labels = result.labels or [f"col{i}" for i in range(len(mean))]
        biggest = int(np.argmax(mean))
        for i, (label, value) in enumerate(zip(labels, mean)):
            mark = "  <--" if i == biggest else ""
            print(f"  {label:<24} {value:10.4f}{mark}")

    print()
    print("The labels are the point. BandPower names each output, so a")
    print("CsvWriter downstream writes a header you can read a year")
    print("later, and a node selecting a feature can ask for it by name")
    print("instead of by column index.")
    print()
    print("Two knobs decide what these numbers mean. The FFT's")
    print("window_size sets the frequency resolution -- 250 samples at")
    print("250 Hz is one second, so bins are 1 Hz apart -- and it also")
    print("sets how often an answer appears at all, which is the delay")
    print("before your application can react.")
    print()
    print("bands= takes explicit ranges too, e.g. bands=[[8, 12]] or a")
    print("mix of names and ranges, when the conventional boundaries")
    print("are not the ones your protocol uses.")
