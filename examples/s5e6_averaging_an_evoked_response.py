"""Season 5, Episode 6: how much averaging removes, measured on noise."""

import time

import numpy as np

import gpype as gp

SAMPLING_RATE = 250
EPOCHS = 40
TIME_PRE = 0.2  # seconds of epoch before the marker
TIME_POST = 0.3  # and after it
# One epoch's length between markers, so no two epochs share a sample:
# overlapping epochs are not independent trials
INTERVAL = TIME_PRE + TIME_POST


def rms(block) -> float:
    """Root mean square over everything in the block."""
    array = np.asarray(block, dtype=float)
    return float(np.sqrt(np.mean(np.square(array))))


if __name__ == "__main__":

    p = gp.Pipeline()

    # Ongoing "EEG": noise and nothing else
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                          signal_amplitude=0.0, noise_amplitude=10.0)

    # Markers fired by this script, standing in for a paradigm
    marker = gp.Marker()

    # An epoch around each marker, its pre-stimulus level subtracted,
    # then averaged as they arrive
    trigger = gp.Trigger(time_pre=TIME_PRE, time_post=TIME_POST, target=1)
    baseline = gp.Baseline()
    average = gp.EpochAverage(mode=gp.EpochAverage.CUMULATIVE)

    # Epochs stack into (time, channel, trial): one trial per epoch, and
    # for the running average one trial per update
    single = gp.Collector(name="single_epochs")
    averaged = gp.Collector(name="average")
    counted = gp.Collector(name="count")

    p.connect(source, trigger)
    p.connect(marker, trigger[gp.Trigger.PORT_TRIGGER])
    p.connect(trigger, baseline)
    p.connect(baseline, single)
    p.connect(baseline, average)
    p.connect(average, averaged)
    p.connect(average[gp.EpochAverage.PORT_COUNT], counted)

    p.start()
    # Let the buffer fill before the first marker, so the first epoch
    # has its pre-stimulus samples
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
    # The last trial of the running average is the final average
    average_rms = (rms(mean.data[..., -1]) if mean is not None
                   else float("nan"))

    print(f"{EPOCHS} markers fired, {n_used} epochs averaged")
    print()
    print(f"  rms across single epochs   {single_rms:8.3f}")
    print(f"  rms of the final average   {average_rms:8.3f}")
    if n_used > 0 and average_rms > 0:
        print(f"  ratio                      {single_rms / average_rms:8.2f}"
              f"   (sqrt({n_used}) = {np.sqrt(n_used):.2f})")
