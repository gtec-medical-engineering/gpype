"""Season 4, Episode 8: a stream from a program that is not g.Pype.

The publisher is the thread below, and nothing in it imports gpype:
raw pylsl pushing noise, labelled NOISE1 to NOISE8. In a real session
it is another vendor's amplifier, MATLAB or a stimulus program, usually
on another machine; delete the thread and keep the rest.
"""

import threading
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet

import gpype as gp

STREAM_NAME = "NoiseAmp"  # what the other program calls itself
CHANNELS = 8
FS = 250
BLOCK = 5  # samples per push, fifty pushes a second


def foreign_publisher(stop: threading.Event) -> None:
    """The other vendor's program. No gpype below this line."""
    info = StreamInfo(name=STREAM_NAME, type="EEG", channel_count=CHANNELS,
                      nominal_srate=FS, channel_format="float32",
                      source_id="s4e8-noise")
    # Channel labels travel in the stream description
    channels = info.desc().append_child("channels")
    for index in range(CHANNELS):
        channels.append_child("channel").append_child_value(
            "label", f"NOISE{index + 1}")
    outlet = StreamOutlet(info, chunk_size=BLOCK)
    rng = np.random.default_rng(7)
    # Paced against a running deadline, so the rate does not drift
    deadline = time.perf_counter()
    while not stop.is_set():
        chunk = rng.normal(0.0, 20.0, size=(BLOCK, CHANNELS))
        outlet.push_chunk(chunk.astype(np.float32).tolist())
        deadline += BLOCK / FS
        time.sleep(max(0.0, deadline - time.perf_counter()))


if __name__ == "__main__":

    # The publisher first: the receiver looks for the stream as soon as
    # it is created, and gives up after five seconds
    stop = threading.Event()
    threading.Thread(target=foreign_publisher, args=(stop,),
                     daemon=True).start()

    app = gp.MainApp()
    p = gp.Pipeline()

    # By name, since a lab network carries more than one stream of type
    # EEG; stream_type="EEG" would take whichever answers first. The
    # channel count and the rate are read from the stream
    source = gp.LslReceiver(stream_name=STREAM_NAME)

    scope = gp.TimeSeriesScope(amplitude_limit=100, time_window=10)

    p.connect(source, scope)
    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()
    stop.set()
