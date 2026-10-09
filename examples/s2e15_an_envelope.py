"""Season 2, Episode 15: a RollingStatistic reads an envelope."""

import time

import numpy as np

import gpype as gp

SAMPLING_RATE = 250
SECONDS = 4.0
WINDOW = 50  # samples: 0.2 s, two cycles of 10 Hz


def collected(collector):
    """The one channel a Collector kept, as float64."""
    return np.asarray(collector.result.data, dtype=np.float64)[:, 0]


def transitions(collector):
    """How many times a Threshold's output changed state."""
    return int(np.count_nonzero(np.diff(collected(collector))))


if __name__ == "__main__":

    p = gp.Pipeline()

    # A 10 Hz carrier switched on and off once a second, plus noise that
    # is there all the time. frame_size=1 keeps every node at 250 Hz:
    # RollingStatistic emits one value per input frame
    carrier = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                           frame_size=1, signal_frequency=10,
                           signal_amplitude=1.0, name="carrier")
    gate = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                        frame_size=1, signal_frequency=0.5,
                        signal_shape="rect", signal_amplitude=1.0,
                        name="gate")
    noise = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                         frame_size=1, noise_amplitude=0.3, name="noise")
    bursts = gp.Equation("a * (1 + g) / 2 + n")
    p.connect(carrier, bursts["a"])
    p.connect(gate, bursts["g"])
    p.connect(noise, bursts["n"])

    # The envelope: the RMS over the last WINDOW samples, one per sample
    envelope = gp.RollingStatistic(statistic="rms", window_size=WINDOW)
    p.connect(bursts, envelope)

    # Three decisions: on the signal, on its envelope, and on the envelope
    # debounced as Episode 12 taught
    on_signal = gp.Threshold(level=0.5)
    on_envelope = gp.Threshold(level=0.5)
    debounced = gp.Threshold(level=0.55, release=0.45, dwell=25)
    p.connect(bursts, on_signal)
    p.connect(envelope, on_envelope)
    p.connect(envelope, debounced)

    kept = {name: gp.Collector(name=name)
            for name in ("envelope", "on_signal", "on_envelope",
                         "debounced")}
    p.connect(envelope, kept["envelope"])
    p.connect(on_signal, kept["on_signal"])
    p.connect(on_envelope, kept["on_envelope"])
    p.connect(debounced, kept["debounced"])

    p.start()
    time.sleep(SECONDS)
    p.stop()
    p.close()

    # The first burst is samples 0-249, the first pause 250-499: read
    # each well away from its edges, once the window has filled
    rms = collected(kept["envelope"])
    print(f"{SECONDS:.0f} s: a 10 Hz burst every other second, amplitude 1, "
          "noise of standard deviation 0.3")
    print(f"  envelope during a burst   {np.median(rms[100:240]):.2f}")
    print(f"  envelope between bursts   {np.median(rms[350:490]):.2f}")
    print(f"  Threshold(level=0.5) on the signal      "
          f"{transitions(kept['on_signal']):5d} transitions")
    print(f"  Threshold(level=0.5) on the envelope    "
          f"{transitions(kept['on_envelope']):5d} transitions")
    print(f"  ... with release=0.45, dwell=25         "
          f"{transitions(kept['debounced']):5d} transitions")
