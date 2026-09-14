# --------------------------------------------------------------
# Example file s5e6_averaging_an_evoked_response.py
# For details and usage, see g.Pype Training Season 5, Episode 6
# --------------------------------------------------------------
#
# An evoked response is small. It is buried perhaps ten or a hundred
# times below the ongoing EEG, which is why nobody looks at a single
# trial: you average many, and the response emerges because it is the
# same every time while everything else is not.
#
# That is a claim you can measure, and this episode measures it. The
# generator here produces *only* noise -- no evoked response at all --
# so whatever survives averaging is exactly the part that should shrink.
# If averaging works, the average of N epochs has about sqrt(N) times
# less of it than a single epoch does.
#
# The nodes involved are the three an ERP pipeline always uses:
#
#   Trigger      cuts an epoch around each marker
#   Baseline     subtracts each epoch's pre-stimulus level
#   EpochAverage averages them as they arrive, and says how many
#
# Baseline matters more than it looks. Without it a slow drift between
# trials survives averaging as an offset, because each epoch starts
# wherever the drift happened to be rather than at a common zero.

import time

import numpy as np

import gpype as gp

SAMPLING_RATE = 250
EPOCHS = 40
INTERVAL = 0.1          # seconds between markers
TIME_PRE = 0.2          # seconds of epoch before the marker
TIME_POST = 0.3         # and after it


def epoch_collector(name: str) -> "gp.Collector":
    """A Collector that accepts epochs rather than a stream.

    An epoch is not a continuous signal: it arrives when a trigger
    fires, not on the sampling clock, so the ports that carry it are
    declared Async. A plain Collector's input is Sync and refuses the
    connection by name -- which is the pipeline telling you the two
    kinds of data are not interchangeable, not an obstacle to work
    around. Declaring the port is how you say you meant it.
    """
    return gp.Collector(
        name=name,
        input_ports=[
            gp.IPort.Configuration(
                name=gp.Constants.Defaults.PORT_IN,
                timing=gp.Constants.Timing.ASYNC,
            )
        ],
    )


def rms(block) -> float:
    array = np.asarray(block, dtype=float)
    return float(np.sqrt(np.mean(np.square(array))))


if __name__ == "__main__":

    p = gp.Pipeline()

    # Ongoing "EEG": noise and nothing else.
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                          signal_amplitude=0.0, noise_amplitude=10.0)

    # Markers fired by this script, standing in for a paradigm.
    marker = gp.Marker()

    trigger = gp.Trigger(time_pre=TIME_PRE, time_post=TIME_POST, target=1)
    average = gp.EpochAverage(mode=gp.EpochAverage.CUMULATIVE)

    single = epoch_collector("single_epochs")
    averaged = epoch_collector("average")
    counted = epoch_collector("count")

    baseline = gp.Baseline()
    p.connect(source, trigger)
    p.connect(marker, trigger[gp.Trigger.PORT_TRIGGER])
    p.connect(trigger, baseline)
    p.connect(baseline, single)
    p.connect(baseline, average)
    p.connect(average, averaged)
    p.connect(average[gp.EpochAverage.PORT_COUNT], counted)

    p.start()
    # Let the buffer fill before the first marker, so the first epoch
    # has its pre-stimulus samples.
    time.sleep(TIME_PRE + 0.3)
    for _ in range(EPOCHS):
        marker.emit(1)
        time.sleep(INTERVAL)
    time.sleep(TIME_POST + 0.3)
    p.stop()
    p.close()

    epochs = single.result
    mean = averaged.result
    counts = counted.result

    n_used = int(np.max(counts.data)) if counts is not None else 0
    single_rms = rms(epochs.data) if epochs is not None else float("nan")
    average_rms = rms(mean.data[-1:]) if mean is not None else float("nan")

    print()
    print(f"{EPOCHS} markers fired, {n_used} epochs averaged")
    print()
    print(f"  rms across single epochs   {single_rms:8.3f}")
    print(f"  rms of the final average   {average_rms:8.3f}")
    if n_used > 0 and average_rms > 0:
        print(f"  ratio                      {single_rms / average_rms:8.2f}"
              f"   (sqrt({n_used}) = {np.sqrt(n_used):.2f})")
    print()
    print("There was no evoked response in this signal at all. The")
    print("generator produced noise and nothing else, so the average is")
    print("measuring only how much of the noise averaging removes -- and")
    print("the ratio lands near the square root of the number of epochs,")
    print("which is the whole reason the technique works.")
    print()
    print("Read it the other way round and it becomes a planning tool.")
    print("Averaging buys signal-to-noise as sqrt(N), so going from 40")
    print("trials to 160 is not four times better, it is two. That is")
    print("the arithmetic behind every protocol that asks a subject to")
    print("sit through hundreds of repetitions.")
    print()
    print("Three notes on the nodes.")
    print()
    print("Trigger needs its buffer filled before the first marker,")
    print("which is why this script waits before firing. A marker that")
    print("arrives sooner than time_pre after the start has no")
    print("pre-stimulus data to cut.")
    print()
    print("Baseline sits between the Trigger and the average, and it")
    print("earns its place: each epoch's pre-stimulus level is")
    print("subtracted, so every trial starts from a common zero.")
    print("Without it a slow drift between trials survives averaging")
    print("as an offset, and an ERP measured against a moving baseline")
    print("is not measured against anything.")
    print()
    print("EpochAverage has a second output, count, which says how many")
    print("epochs went into the current average. Record it: an average")
    print("without its N is not a result anyone can interpret, and it")
    print("also tells you how many epochs the rejection thresholds")
    print("(reject_level, reject_peak_to_peak) threw away.")
