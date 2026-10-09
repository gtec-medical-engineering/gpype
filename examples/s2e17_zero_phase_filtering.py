"""Season 2, Episode 17: a bandpass causal and zero-phase, offline only."""

import os
import tempfile

import numpy as np

import gpype as gp

RECORDING = os.path.join(tempfile.gettempdir(), "s2e17_burst.csv")
RATE = 250.0
SAMPLES = 1000
CENTRE = 500


def write_recording(path=RECORDING):
    """Write a burst that is symmetric about its centre sample."""
    index = np.arange(SAMPLES)
    signal = np.exp(-(((index - CENTRE) / 8.0) ** 2)) * np.cos(
        2 * np.pi * 10.0 * (index - CENTRE) / RATE)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("Time, Ch01\n")
        for i in range(SAMPLES):
            handle.write(f"{i / RATE:g}, {signal[i]:.10f}\n")
    return signal


def filtered(path, phase):
    """Bandpass the recording offline, causal or zero-phase."""
    with gp.Pipeline() as p:
        reader = gp.CsvReader(file_name=path, mode="batch")
        bandpass = gp.Bandpass(f_lo=1, f_hi=30, phase=phase)
        collector = gp.Collector()
        p.connect(reader, bandpass)
        p.connect(bandpass, collector)
        return p.run().data[:, 0].astype(np.float64)


def asymmetry(trace):
    """How far a trace is from symmetric about the centre, 0 to 1."""
    window = trace[CENTRE - 200:CENTRE + 201]
    scale = np.max(np.abs(window)) or 1.0
    return float(np.max(np.abs(window - window[::-1])) / scale)


def lag(trace, reference):
    """The lag in samples that best aligns a trace with the input."""
    a = trace - trace.mean()
    b = reference - reference.mean()
    return int(np.argmax(np.correlate(a, b, mode="full")) - (len(b) - 1))


if __name__ == "__main__":

    signal = write_recording()
    for phase in ("causal", "zero"):
        trace = filtered(RECORDING, phase)
        shift = lag(trace, signal)
        print(f"{phase:7}: asymmetry {asymmetry(trace):.3f}   "
              f"lag {shift:+d} samples ({shift / RATE * 1000:+.1f} ms)")

    # Against the clock the zero-phase form is refused: it would need
    # samples that have not arrived yet
    with gp.Pipeline() as p:
        source = gp.Generator(sampling_rate=RATE, channel_count=1)
        p.connect(source, gp.Bandpass(f_lo=1, f_hi=30, phase="zero"))
        try:
            p.start()
        except RuntimeError as refusal:
            print(f"realtime: {refusal}")
        else:
            p.stop()
