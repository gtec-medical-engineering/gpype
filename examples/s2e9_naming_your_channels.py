# --------------------------------------------------------------
# Example file s2e9_naming_your_channels.py
# For details and usage, see g.Pype Training Season 2, Episode 9
# --------------------------------------------------------------
#
# Most sources carry no channel names at all. That is honest -- a
# generator does not know where you put the electrodes -- but it means
# nothing downstream can say more than "channel 3". A CsvWriter handed a
# nameless stream invents Ch01, Ch02, ... for its header, and that is
# what the recording is stuck with afterwards.
#
# ChannelLabeler names them once, near the source, and the names travel
# with the data from there: into scopes, into recordings, and into any
# node that selects a channel by name rather than by number.

import time

import gpype as gp

SAMPLING_RATE = 250
ELECTRODES = ["Fz", "C3", "Cz", "C4", "P3", "Pz", "P4", "Oz"]


def labels_of(collector) -> list:
    """The channel names a collector saw, or an empty list."""
    result = collector.result
    return list(result.labels) if result and result.labels else []


if __name__ == "__main__":

    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=8,
                          signal_amplitude=1.0)

    before = gp.Collector(name="before")
    after = gp.Collector(name="after")

    # A Montage is a list of electrode names with an optional reference.
    # standard_1020 additionally insists that every name is a real 10-20
    # position, so a typo is refused here rather than ending up in a
    # recording that outlives the experiment.
    montage = gp.Montage.standard_1020(ELECTRODES, reference="Cz")

    labeler = gp.ChannelLabeler(montage=montage, name="labeler")

    p.connect(source, before)
    p.connect(source, labeler)
    p.connect(labeler, after)

    p.start()
    time.sleep(0.5)
    p.stop()
    p.close()

    print()
    print("before ChannelLabeler:", labels_of(before))
    print("after  ChannelLabeler:", labels_of(after))
    print()
    print("An empty list is not a bug: the generator genuinely has no")
    print("names to give. What it means is that anything downstream has")
    print("to invent them -- a CsvWriter writes Ch01, Ch02, ... -- and")
    print("the recording then carries invented names for good.")
    print()
    print("So name channels once, as early as you can. Everything after")
    print("the labeler inherits the names: a scope draws them on its")
    print("axis, a CsvWriter puts them in the header, and a node that")
    print("selects by name can find 'Cz' instead of counting to three.")
    print()
    print("A Montage is not required -- ChannelLabeler also takes a")
    print("plain labels=[...] list. The montage is worth it when the")
    print("names are electrodes, because standard_1020 refuses a name")
    print("that is not a real position, and a typo caught here is a")
    print("typo that never reaches the recording:")
    print()
    try:
        gp.Montage.standard_1020(["Fz", "Cz", "Zz"])
    except ValueError as error:
        print(" ", error)
    print()
    print("Note the set it knows: the classic 10-20 positions. Denser")
    print("10-10 names such as PO7 are not in it, so an electrode set")
    print("that uses them takes the plain labels=[...] route instead.")
