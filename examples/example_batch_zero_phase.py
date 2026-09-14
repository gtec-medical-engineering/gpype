"""Compare causal and zero-phase filtering of the same recording.

Zero phase is what a realtime pipeline cannot do: it filters forwards and
backwards, so it needs the whole recording. Measured here by symmetry and
cross-correlation, not by peak position -- both filters put the maximum on
the same sample. The last section shows the same filter being refused in a
realtime pipeline. Writes its test recording to the current directory.

Run: python example_batch_zero_phase.py
"""

import time

import numpy as np

import gpype as gp

#: The recording this example writes and then reads back.
RECORDING = "example_batch_burst.csv"

RATE = 250.0
SAMPLES = 1000
CENTRE = 500


def write_recording(path: str = RECORDING) -> np.ndarray:
    """Write a burst that is symmetric about its centre sample.

    Args:
        path: File to write.

    Returns:
        The signal that was written, for comparison later.
    """
    index = np.arange(SAMPLES)
    signal = np.exp(-(((index - CENTRE) / 8.0) ** 2)) * np.cos(
        2 * np.pi * 10.0 * (index - CENTRE) / RATE
    )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("Time, Ch01\n")
        for i in range(SAMPLES):
            handle.write(f"{i / RATE:g}, {signal[i]:.10f}\n")
    return signal


def filtered(path: str, phase: str) -> np.ndarray:
    """Bandpass a recording offline, one way or the other.

    Args:
        path: The recording to read.
        phase: ``causal`` or ``zero``.

    Returns:
        The filtered channel.
    """
    with gp.Pipeline() as p:
        reader = gp.CsvReader(file_name=path, mode="batch")
        bandpass = gp.Bandpass(f_lo=1, f_hi=30, phase=phase)
        collector = gp.Collector()
        p.connect(reader, bandpass)
        p.connect(bandpass, collector)
        return p.run().data[:, 0].astype(np.float64)


def asymmetry(trace: np.ndarray) -> float:
    """How far a trace is from symmetric about the centre, 0 to 1."""
    window = trace[CENTRE - 200 : CENTRE + 201]
    scale = np.max(np.abs(window)) or 1.0
    return float(np.max(np.abs(window - window[::-1])) / scale)


def lag(trace: np.ndarray, reference: np.ndarray) -> int:
    """Lag in samples that best aligns a trace with the input."""
    a = trace - trace.mean()
    b = reference - reference.mean()
    return int(np.argmax(np.correlate(a, b, mode="full")) - (len(b) - 1))


if __name__ == "__main__":

    signal = write_recording()

    traces = {
        phase: filtered(RECORDING, phase) for phase in ("causal", "zero")
    }

    for phase, trace in traces.items():
        shift = lag(trace, signal)
        print(
            f"{phase:7}: asymmetry {asymmetry(trace):.3f}   "
            f"lag {shift:+d} samples ({shift / RATE * 1000:+.1f} ms)"
        )

    # The refusal is part of the design, not an accident: a zero-phase
    # filter in a realtime pipeline would emit nothing, forever, so the node
    # rejects it while being set up. This is what that looks like from
    # the outside -- an ordinary realtime pipeline, started the ordinary
    # way, which reports the failure through its monitor.
    print()
    print("Against the clock, the same filter is refused:")
    with gp.Pipeline() as p:
        source = gp.Generator(
            sampling_rate=RATE, channel_count=1, frame_size=10
        )
        p.connect(source, gp.Bandpass(f_lo=1, f_hi=30, phase="zero"))
        # start() refuses a pipeline that cannot work and raises there
        # and then, naming the node. Catching it is the point of this
        # block: uncaught, the refusal this example exists to show would
        # end the script before it drew anything.
        try:
            p.start()
        except RuntimeError as refusal:
            print(f"  {refusal}")
        else:
            p.stop()
            print("  ...which did not happen; the filter was accepted.")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nInstall matplotlib to see the plot.")
    else:
        figure, axes = plt.subplots(figsize=(9, 4))
        times = np.arange(SAMPLES) / RATE
        axes.plot(times, signal, color="0.7", label="input")
        axes.plot(times, traces["causal"], label="causal (group delay)")
        axes.plot(times, traces["zero"], label="zero phase")
        axes.set_xlim(1.6, 2.4)
        axes.set_xlabel("time (s)")
        axes.set_title("The same 1-30 Hz bandpass, both ways")
        axes.legend(loc="upper right", fontsize="small")
        plt.tight_layout()
        plt.show()
