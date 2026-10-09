"""Season 7, Episode 1: a node of your own: parameters, setup(), step()."""

import os
import tempfile
import time

import numpy as np

import gpype as gp
from gpype.common.document import with_fresh_ids

KEYS = gp.Constants.Keys
ROLES = gp.Constants.ChannelRoles
PORT_IN = gp.Constants.Defaults.PORT_IN
PORT_OUT = gp.Constants.Defaults.PORT_OUT

RATE = 250
SECONDS = 8
WINDOW = 1.5
RECORDING = os.path.join(tempfile.gettempdir(), "s7e1_your_first_node.csv")


class DriftMonitor(gp.IONode):
    """Each channel's distance from its running mean, and the worst."""

    class Configuration(gp.IONode.Configuration):
        class Keys(gp.IONode.Configuration.Keys):
            #: Required: None or an empty value is refused.
            WINDOW_SECONDS = "window_seconds"

        class OptionalKeys(gp.IONode.Configuration.OptionalKeys):
            #: May be left out.
            LABEL = "label"

    def __init__(self, window_seconds: float = 2.0, label: str = None,
                 **kwargs):
        # What reaches the base class is the node's configuration, and
        # the configuration is what a saved pipeline stores.
        super().__init__(window_seconds=window_seconds, label=label,
                         **kwargs)
        self._mean = None
        self._weight = None

    def setup(self, data, port_context_in):
        # The base class checks the input and copies its context to
        # every output port. What changes is ours to say.
        port_context_out = super().setup(data, port_context_in)
        context = port_context_in[PORT_IN]
        out = port_context_out[PORT_OUT]

        # The window is in seconds and a frame is in samples. The rate
        # that converts one into the other is first known here.
        window = self.config[self.Configuration.Keys.WINDOW_SECONDS]
        self._weight = 1.0 / max(1.0, context[KEYS.SAMPLING_RATE] * window)
        count = context[KEYS.CHANNEL_COUNT]
        self._mean = np.zeros(count, dtype=gp.Constants.DATA_TYPE)

        # One channel more than came in, and its description with it.
        # Auxiliary, so a filter downstream leaves it alone.
        out[KEYS.CHANNEL_COUNT] = count + 1
        roles = context.get(KEYS.CHANNEL_ROLES) or [ROLES.SIGNAL] * count
        out[KEYS.CHANNEL_ROLES] = list(roles) + [ROLES.AUXILIARY]
        labels = context.get(KEYS.CHANNEL_LABELS)
        if labels:
            label = self.config[self.Configuration.OptionalKeys.LABEL]
            out[KEYS.CHANNEL_LABELS] = list(labels) + [label or "worst"]
        return port_context_out

    def step(self, data):
        frame = data[PORT_IN]
        drift = np.empty_like(frame)
        for i, row in enumerate(frame):
            self._mean += self._weight * (row - self._mean)
            drift[i] = row - self._mean
        worst = np.max(np.abs(drift), axis=1, keepdims=True)
        return {PORT_OUT: np.hstack((drift, worst))}


class RecordDrift(DriftMonitor):
    """The same measurement against the mean of the whole recording.

    That mean exists only once the recording does. A batch run calls
    process() once, with every sample; a realtime run would call the
    step() this class inherits.
    """

    def process(self, data):
        record = data[PORT_IN]
        drift = record - record.mean(axis=0)
        worst = np.max(np.abs(drift), axis=1, keepdims=True)
        return {PORT_OUT: np.hstack((drift, worst))}


def write_recording(path: str = RECORDING) -> None:
    """Eight seconds of three channels; Cz drifts up by 40 over them."""
    t = np.arange(RATE * SECONDS) / RATE
    rng = np.random.default_rng(7)
    data = 5.0 * np.sin(2 * np.pi * 10 * t)[:, None] + rng.normal(
        0.0, 1.0, (t.size, 3))
    data[:, 1] += 40.0 * t / SECONDS
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("Time, Fz, Cz, Pz\n")
        for second, row in zip(t, data):
            handle.write(f"{second:g}, " + ", ".join(f"{v:.6f}" for v in row)
                         + "\n")


def replay(node) -> tuple:
    """Frame by frame, as fast as the machine allows.

    Returns:
        What the Collector kept, and the pipeline, serialized.
    """
    with gp.Pipeline() as p:
        reader = gp.CsvReader(file_name=RECORDING, frame_size=10, speed=0)
        collector = gp.Collector()
        p.connect(reader, node)
        p.connect(node, collector)
        p.start()
        # Wait for the Collector, not the reader: a reader is exhausted
        # once its last frame has left it, before that frame arrives.
        while (collector.result is None
               or len(collector.result.data) < RATE * SECONDS):
            time.sleep(0.01)
        p.stop()
        return collector.result, p.serialize()


def offline(node) -> gp.Result:
    """The whole recording in one cycle."""
    with gp.Pipeline() as p:
        reader = gp.CsvReader(file_name=RECORDING, mode="batch")
        collector = gp.Collector()
        p.connect(reader, node)
        p.connect(node, collector)
        return p.run()


if __name__ == "__main__":

    write_recording()

    # 1. Frame by frame. The output is one channel wider than the input,
    #    and its labels and roles say so.
    live, document = replay(DriftMonitor(window_seconds=WINDOW))
    print(f"out: {live.data.shape[1]} channels")
    print(f"labels: {live.labels}")
    print(f"roles:  {live.roles}")

    # 2. The saved pipeline carries the parameter, and loading it builds
    #    a node that has it. Loading imports the module each node names,
    #    so a module of your own has to be allowed first (Season 8). The
    #    first pipeline's nodes still hold their ids in this process, so
    #    the copy is given new ones.
    saved = next(n for n in document["nodes"] if n["class"] == "DriftMonitor")
    print(f"saved:  window_seconds={saved['config']['window_seconds']}, "
          f"label={saved['config']['label']}")
    gp.allow_modules(DriftMonitor.__module__)
    loaded = gp.Pipeline.deserialize(with_fresh_ids(document))
    again = next(n for n in loaded.serialize()["nodes"]
                 if n["class"] == "DriftMonitor")
    print(f"loaded: window_seconds={again['config']['window_seconds']}, "
          f"label={again['config']['label']}")
    loaded.close()

    # A key in Keys has to be given; one in OptionalKeys may be left out.
    try:
        DriftMonitor(window_seconds=None)
    except ValueError as error:
        print(f"refused: {error}")

    # 3. Offline. step() is causal, so the whole recording in one call
    #    gives exactly what the frames gave. process() is for what only
    #    a whole recording can answer.
    batch = offline(DriftMonitor(window_seconds=WINDOW))
    record = offline(RecordDrift(window_seconds=WINDOW))
    same = np.array_equal(batch.data, live.data)
    print(f"step() offline equals frame by frame: {same}")
    print(f"Cz at the end, step():    {batch.data[-RATE:, 1].mean():5.1f}")
    print(f"Cz at the end, process(): {record.data[-RATE:, 1].mean():5.1f}")
