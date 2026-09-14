"""Record five seconds to CSV, then reprocess that recording offline.

Both halves run, so the relationship is visible: the Bandpass applied to
the whole file is the same node, with the same coefficients, that ran
against the clock. Pipeline.run() returns when the last node is done, so
there is no stop() to time. The recording is written to the current
directory and reused on the next run.

Run: python example_batch_reprocess.py
"""

import glob
import os
import time

import gpype as gp

#: Base name for the recording. CsvWriter appends a timestamp.
RECORDING = "example_batch_recording.csv"


def record(seconds: float = 5.0) -> str:
    """Record from the synthetic generator, paced against the clock.

    An ordinary realtime pipeline: start it, let it run, stop it. No
    widgets, so it needs no Qt and no display.

    Args:
        seconds: How long to record.

    Returns:
        Path of the file that was written.
    """
    p = gp.Pipeline()
    source = gp.Generator(
        sampling_rate=250,
        channel_count=4,
        signal_frequency=10,
        signal_amplitude=10,
        noise_amplitude=2,
        frame_size=10,
    )
    writer = gp.CsvWriter(file_name=RECORDING)
    p.connect(source, writer)

    print(f"Recording {seconds:g} s ...")
    p.start()
    time.sleep(seconds)
    p.stop()
    p.close()

    written = glob.glob("example_batch_recording*.csv")
    return max(written, key=os.path.getmtime)


def reprocess(path: str):
    """Filter a whole recording in one pass.

    Args:
        path: The recording to read.

    Returns:
        The Result the Collector kept.
    """
    with gp.Pipeline() as p:
        # mode="batch" makes the frame the whole recording, so the
        # reader emits it in a single cycle and the run has an end.
        reader = gp.CsvReader(file_name=path, mode="batch")

        # The same node a realtime pipeline would use, unchanged.
        bandpass = gp.Bandpass(f_lo=9, f_hi=11)

        # Terminates the pipeline in memory rather than in a second file.
        collector = gp.Collector()

        p.connect(reader, bandpass)
        p.connect(bandpass, collector)

        # Blocking: returns when the last node has finished.
        return p.run()


if __name__ == "__main__":

    # Reuse a recording if one is lying about, so a second run is quick.
    existing = sorted(glob.glob("example_batch_recording*.csv"))
    path = existing[-1] if existing else record()
    print(f"Reprocessing {path}")

    result = reprocess(path)

    # The Result says what it is, rather than being a bare array.
    print()
    print(result)
    print(f"  shape    : {result.data.shape}")
    print(f"  rate     : {result.rate} Hz")
    print(f"  duration : {result.duration:.2f} s")
    print(f"  labels   : {result.labels}")
    print(f"  roles    : {result.roles}")

    # ... and it is a numpy array underneath, for anything else.
    channel_1 = result.data[:, 0]
    print()
    print(
        f"  peak-to-peak of channel 1: "
        f"{channel_1.max() - channel_1.min():.2f}"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nInstall matplotlib to see the plot.")
    else:
        axes = result.plot()
        axes.set_title(f"9-11 Hz bandpass, offline ({os.path.basename(path)})")
        axes.set_ylabel(
            f"amplitude ({result.units[0]})" if result.units else "amplitude"
        )
        axes.legend(loc="upper right", fontsize="small")
        plt.tight_layout()
        plt.show()
