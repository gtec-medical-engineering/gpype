"""Season 7, Episode 6: a plain function as a batch node, two ways.

A transform that is one line of numpy needs no node class. gp.Apply(fn)
is the node, and @gp.node turns the function into a factory for one
while leaving it callable on its own. Both are batch only: a realtime
pipeline refuses them at start(), naming the node.
"""

import os
import tempfile

import numpy as np

import gpype as gp

#: The recording this example writes and then reads back.
RECORDING = os.path.join(tempfile.gettempdir(),
                         "s7e6_a_function_as_a_node.csv")

RATE = 250.0
SAMPLES = 500


def write_recording(path: str = RECORDING) -> None:
    """Write two seconds of a 10 Hz sine, amplitude 20.

    Args:
        path: File to write.
    """
    times = np.arange(SAMPLES) / RATE
    signal = 20.0 * np.sin(2 * np.pi * 10.0 * times)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("Time, Ch01\n")
        for time, value in zip(times, signal):
            handle.write(f"{time:g}, {value:.10f}\n")


def rectify(block: np.ndarray) -> np.ndarray:
    """Full-wave rectification: the magnitude of every sample."""
    return np.abs(block)


@gp.node
def rectified(block: np.ndarray) -> np.ndarray:
    """The same function, decorated: rectified() builds an Apply."""
    return np.abs(block)


def run(node) -> gp.Result:
    """Read the recording through *node* in one batch run.

    Args:
        node: The node to put between the reader and the Collector.

    Returns:
        What the Collector kept.
    """
    with gp.Pipeline() as p:
        reader = gp.CsvReader(file_name=RECORDING, mode="batch")
        collector = gp.Collector()
        p.connect(reader, node)
        p.connect(node, collector)
        return p.run()


if __name__ == "__main__":

    write_recording()

    # 1. Apply takes any callable: one array of (samples, channels) in,
    #    one of the same shape and dtype out.
    explicit = run(gp.Apply(rectify, name="rectify"))

    # 2. The decorated form. rectified() is an Apply named after the
    #    function; rectified.function is the function itself.
    decorated = run(rectified())

    print(f"Apply(rectify)   : {explicit.data.shape}, "
          f"min {explicit.data.min():.2f}, max {explicit.data.max():.2f}")
    print(f"@gp.node         : {decorated.data.shape}, "
          f"identical: {np.array_equal(explicit.data, decorated.data)}")

    # 3. The function stays testable with no pipeline anywhere near it.
    probe = np.array([[-3.0], [0.0], [2.0]], dtype=np.float32)
    print(f"rectified.function({probe[:, 0].tolist()}) = "
          f"{rectified.function(probe)[:, 0].tolist()}")

    # 4. Against the clock, the same node is refused. A function has no
    #    bound on how long it takes, and a realtime frame has a deadline.
    print()
    print("Against the clock, Apply is refused:")
    with gp.Pipeline() as p:
        source = gp.Generator(sampling_rate=RATE, channel_count=1)
        p.connect(source, gp.Apply(rectify, name="rectify"))
        try:
            p.start()
        except Exception as refusal:
            print(f"  {type(refusal).__name__}: {refusal}")
        else:
            p.stop()
            print("  ...which did not happen; the node was accepted.")
