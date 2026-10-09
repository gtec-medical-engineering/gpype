"""Season 2, Episode 16: record five seconds, then process them offline."""

import glob
import os
import tempfile
import time

import gpype as gp

# The recording goes to the temp directory; CsvWriter adds a timestamp
RECORDING = os.path.join(tempfile.gettempdir(), "s2e16_recording.csv")
PATTERN = os.path.join(tempfile.gettempdir(), "s2e16_recording*.csv")


def record(seconds=5.0):
    """Record the generator against the clock, and return the file."""
    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=250, channel_count=4,
                          signal_frequency=10, signal_amplitude=10,
                          noise_amplitude=2)
    writer = gp.CsvWriter(file_name=RECORDING)
    p.connect(source, writer)

    p.start()
    time.sleep(seconds)
    p.stop()
    p.close()
    return max(glob.glob(PATTERN), key=os.path.getmtime)


def reprocess(path):
    """Bandpass the whole recording in one pass, and return the Result."""
    with gp.Pipeline() as p:
        # mode="batch": the whole file is one frame, and the run has an end
        reader = gp.CsvReader(file_name=path, mode="batch")
        # The same node a realtime pipeline uses, unchanged
        bandpass = gp.Bandpass(f_lo=9, f_hi=11)
        # Keeps the result in memory instead of a second file
        collector = gp.Collector()
        p.connect(reader, bandpass)
        p.connect(bandpass, collector)
        # Returns when the last node is done
        return p.run()


if __name__ == "__main__":

    existing = sorted(glob.glob(PATTERN))
    path = existing[-1] if existing else record()
    result = reprocess(path)

    print(f"file     : {os.path.basename(path)}")
    print(f"shape    : {result.data.shape}")
    print(f"rate     : {result.rate} Hz")
    print(f"duration : {result.duration:.2f} s")
    print(f"labels   : {result.labels}")
    channel = result.data[:, 0]
    print(f"peak-to-peak of channel 1: {channel.max() - channel.min():.2f}")
