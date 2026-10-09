"""Season 2, Episode 10: a Reference decides what zero means."""

import time

import numpy as np

import gpype as gp

SAMPLING_RATE = 250
ELECTRODES = ["Fz", "C3", "Cz", "C4", "P3", "Pz", "P4", "Oz"]


def rms(collector):
    """Root mean square of the second half of what was collected."""
    result = collector.result
    if result is None:
        return float("nan")
    block = result.data
    return float(np.sqrt(np.mean(np.square(block[len(block) // 2:]))))


def channels(collector):
    result = collector.result
    return 0 if result is None else result.channel_count


if __name__ == "__main__":

    p = gp.Pipeline()

    # The same 10 Hz sine on every channel, independent noise on each: what
    # all electrodes share is usually not brain
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=8,
                          signal_frequency=10, signal_amplitude=1.0,
                          noise_amplitude=0.2)

    # Named first (Episode 9), so the reference can be chosen by electrode
    labeler = gp.ChannelLabeler(labels=ELECTRODES)

    # No argument: the common average. A label: that electrode
    car = gp.Reference()
    cz = gp.Reference(reference="Cz")

    as_recorded = gp.Collector(name="as_recorded")
    common_average = gp.Collector(name="common_average")
    against_cz = gp.Collector(name="against_Cz")

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

    print(f"as recorded        rms = {rms(as_recorded):6.3f}"
          f"   channels = {channels(as_recorded)}")
    print(f"common average     rms = {rms(common_average):6.3f}"
          f"   channels = {channels(common_average)}")
    print(f"referenced to Cz   rms = {rms(against_cz):6.3f}"
          f"   channels = {channels(against_cz)}")
