"""Season 2, Episode 9: a ChannelLabeler names the channels at the source."""

import time

import gpype as gp

SAMPLING_RATE = 250
ELECTRODES = ["Fz", "C3", "Cz", "C4", "P3", "Pz", "P4", "Oz"]


def labels_of(collector):
    """The channel names a collector saw, or an empty list."""
    result = collector.result
    return list(result.labels) if result and result.labels else []


if __name__ == "__main__":

    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=8,
                          signal_amplitude=1.0)
    before = gp.Collector(name="before")
    after = gp.Collector(name="after")

    # A Montage is a list of electrode names, with an optional reference.
    # standard_1020 refuses a name that is not a real 10-20 position
    montage = gp.Montage.standard_1020(ELECTRODES, reference="Cz")
    labeler = gp.ChannelLabeler(montage=montage, name="labeler")

    p.connect(source, before)
    p.connect(source, labeler)
    p.connect(labeler, after)

    p.start()
    time.sleep(0.5)
    p.stop()
    p.close()

    print("before ChannelLabeler:", labels_of(before))
    print("after  ChannelLabeler:", labels_of(after))

    # A typo is caught here, not in a recording that outlives the experiment
    try:
        gp.Montage.standard_1020(["Fz", "Cz", "Zz"])
    except ValueError as error:
        print("refused:", error)
