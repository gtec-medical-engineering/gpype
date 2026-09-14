# --------------------------------------------------------------
# Example file s2e10_choosing_a_reference.py
# For details and usage, see g.Pype Training Season 2, Episode 10
# --------------------------------------------------------------
#
# An EEG channel is not a measurement of one electrode. It is a voltage
# difference between that electrode and a reference, so choosing the
# reference is choosing what "zero" means -- and it changes every number
# that follows.
#
# The Reference node re-references a recorded stream. This episode shows
# the two choices you will actually make: the common average, and a
# single electrode (or a linked pair).
#
# The demonstration rests on how the generator builds its signal: the
# 10 Hz sine is identical on every channel, while the noise is drawn
# independently per channel. That is a good model of the real thing --
# what all electrodes share is usually not brain -- and it makes the
# effect of a reference measurable in one number.

import time

import numpy as np

import gpype as gp

SAMPLING_RATE = 250
ELECTRODES = ["Fz", "C3", "Cz", "C4", "P3", "Pz", "P4", "Oz"]


def rms(collector) -> float:
    """Root mean square of the second half of what was collected."""
    result = collector.result
    if result is None:
        return float("nan")
    block = result.data
    return float(np.sqrt(np.mean(np.square(block[len(block) // 2:]))))


def channels(collector) -> int:
    result = collector.result
    return 0 if result is None else result.channel_count


if __name__ == "__main__":

    p = gp.Pipeline()

    # Common-mode signal on every channel, plus independent noise.
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=8,
                          signal_frequency=10, signal_amplitude=1.0,
                          noise_amplitude=0.2)

    # Name the channels first (Season 2, Episode 9), so the reference
    # can be chosen by electrode rather than by counting.
    labeler = gp.ChannelLabeler(labels=ELECTRODES)

    as_recorded = gp.Collector(name="as_recorded")
    common_average = gp.Collector(name="common_average")
    against_cz = gp.Collector(name="against_Cz")

    # No argument means the common average: every channel minus the mean
    # over all of them.
    car = gp.Reference()
    # A single label means that electrode becomes the reference.
    cz = gp.Reference(reference="Cz")

    p.connect(source, labeler)
    p.connect(labeler, as_recorded)
    p.connect(labeler, car)
    p.connect(car, common_average)
    p.connect(labeler, cz)
    p.connect(cz, against_cz)

    p.start()
    time.sleep(2.0)
    p.stop()
    p.close()

    print()
    print("A 10 Hz sine shared by all 8 channels, plus independent")
    print("noise of standard deviation 0.2 on each.")
    print()
    print(f"  as recorded          rms = {rms(as_recorded):6.3f}"
          f"   channels = {channels(as_recorded)}")
    print(f"  common average       rms = {rms(common_average):6.3f}"
          f"   channels = {channels(common_average)}")
    print(f"  referenced to Cz     rms = {rms(against_cz):6.3f}"
          f"   channels = {channels(against_cz)}")
    print()
    print("Read those three numbers as three different questions.")
    print()
    print("As recorded is dominated by the shared sine: 0.707 from the")
    print("signal against 0.2 of noise. Whatever every electrode sees")
    print("together is usually not brain, so this is the number you")
    print("least want to analyse.")
    print()
    print("The common average subtracts the mean over all channels, so")
    print("the shared component cancels almost exactly and what is left")
    print("is each channel's own noise. It is the safe default when you")
    print("have enough channels and no reason to prefer a site.")
    print()
    print("Referencing to Cz subtracts one channel from all of them.")
    print("The shared component cancels here too, but Cz's own noise is")
    print("now added to every channel -- which is why this number is")
    print("larger -- and Cz itself becomes a row of zeros, since a")
    print("channel referenced to itself is nothing.")
    print()
    print("There is no neutral choice. There is only a choice you have")
    print("made deliberately and can state in a methods section.")
