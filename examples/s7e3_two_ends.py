"""Season 7, Episode 3: a source and a sink of your own, chains included."""

import os
import sqlite3
import tempfile
import threading
import time

import numpy as np

import gpype as gp

KEYS = gp.Constants.Keys
PORT_IN = gp.Constants.Defaults.PORT_IN
PORT_OUT = gp.Constants.Defaults.PORT_OUT

RATE = 250
SECONDS = 2
DATABASE = os.path.join(tempfile.gettempdir(), "s7e3_two_ends.sqlite")


class Counter(gp.Source):
    """A device that sends a numbered ramp, in whole frames, then stops."""

    class Configuration(gp.Source.Configuration):
        class Keys(gp.Source.Configuration.Keys):
            SAMPLING_RATE = KEYS.SAMPLING_RATE
            SECONDS = "seconds"

    def __init__(self, sampling_rate: float = 250.0, channel_count: int = 2,
                 seconds: float = 2.0, **kwargs):
        output_ports = kwargs.pop(KEYS.OUTPUT_PORTS,
                                  [gp.OPort.Configuration()])
        super().__init__(output_ports=output_ports,
                         channel_count=channel_count,
                         sampling_rate=sampling_rate, seconds=seconds,
                         **kwargs)
        self._thread = None
        self._running = False
        self._frame = None

    def setup(self, data, port_context_in):
        # gp.Source describes the width and the frame size. The rate is
        # this source's to publish, and nothing downstream runs without.
        port_context_out = super().setup(data, port_context_in)
        port_context_out[PORT_OUT][KEYS.SAMPLING_RATE] = (
            self.config[KEYS.SAMPLING_RATE])
        return port_context_out

    def start(self):
        super().start()
        self._running = True
        self._thread = threading.Thread(target=self._acquire, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
        super().stop()

    def _acquire(self):
        # A driver would wait for the device here; this one waits for
        # the clock. A source drives its own cycles: one per frame.
        rate = self.config[KEYS.SAMPLING_RATE]
        size = self.config[KEYS.FRAME_SIZE][0]
        width = self.config[KEYS.CHANNEL_COUNT][0]
        seconds = self.config[self.Configuration.Keys.SECONDS]
        frames = int(rate * seconds) // size
        due = time.perf_counter()
        for first in range(0, frames * size, size):
            if not self._running:
                break
            due += size / rate
            time.sleep(max(0.0, due - time.perf_counter()))
            count = np.arange(first, first + size, dtype=np.float32)
            self._frame = np.repeat(count[:, None], width, axis=1)
            self.cycle()

    def step(self, data):
        frame, self._frame = self._frame, None
        return None if frame is None else {PORT_OUT: frame}


class SqliteWriter(gp.Sink):
    """One row per sample: its time, then one column per channel."""

    class Configuration(gp.Sink.Configuration):
        class Keys(gp.Sink.Configuration.Keys):
            FILE_NAME = "file_name"

    def __init__(self, file_name: str, **kwargs):
        input_ports = kwargs.pop(KEYS.INPUT_PORTS,
                                 [gp.IPort.Configuration()])
        super().__init__(file_name=file_name, input_ports=input_ports,
                         **kwargs)
        self._db = None
        self.written = 0

    def setup(self, data, port_context_in):
        # A row needs its time, and the table needs its columns.
        context = port_context_in[PORT_IN]
        missing = [key for key in (KEYS.SAMPLING_RATE, KEYS.CHANNEL_COUNT)
                   if context.get(key) is None]
        if missing:
            raise ValueError(
                f"SqliteWriter needs {' and '.join(missing)} in its input "
                f"context; got {sorted(context) or 'none'}.")
        self._rate = context[KEYS.SAMPLING_RATE]
        self.written = 0
        columns = ", ".join(f"ch{i + 1} REAL"
                            for i in range(context[KEYS.CHANNEL_COUNT]))
        # setup() and step() run on the source's thread, stop() on the
        # caller's.
        self._db = sqlite3.connect(self.config["file_name"],
                                   check_same_thread=False)
        self._db.execute("DROP TABLE IF EXISTS samples")
        self._db.execute(f"CREATE TABLE samples (t REAL, {columns})")
        return super().setup(data, port_context_in)

    def step(self, data):
        frame = data[PORT_IN]
        t = (self.written + np.arange(len(frame))) / self._rate
        marks = ", ".join("?" * (frame.shape[1] + 1))
        self._db.executemany(f"INSERT INTO samples VALUES ({marks})",
                             np.column_stack((t, frame)).tolist())
        self.written += len(frame)
        return {}

    def stop(self):
        super().stop()
        if self._db is not None:
            self._db.commit()
            self._db.close()
            self._db = None


if __name__ == "__main__":

    p = gp.Pipeline()
    device = Counter(sampling_rate=RATE, channel_count=2, seconds=SECONDS,
                     name="device")
    database = SqliteWriter(file_name=DATABASE, name="database")
    p.connect(device, database)

    # 1. Each end is one class, and the pipeline built its chain.
    for node in (device, database):
        chain = p.chain_of(node)
        inside = ", ".join(type(n).__name__ for n in chain.internal_nodes)
        print(f"{node.name}: {type(chain).__name__} [{inside}]")

    # 2. Run until the device has sent everything and the last row is in.
    p.start()
    while database.written < RATE * SECONDS:
        time.sleep(0.05)
    p.stop()
    p.close()

    db = sqlite3.connect(DATABASE)
    rows, first, last = db.execute(
        "SELECT COUNT(*), MIN(ch1), MAX(ch1) FROM samples").fetchone()
    db.close()
    print(f"{rows} rows for {RATE * SECONDS} samples sent, "
          f"counting {first:g} to {last:g}")

    # 3. A live source g.Pype ships has a port one channel wider than it
    #    was asked for: the instant each frame reached the host, found
    #    by its role. The Sync in its chain removes it.
    port = gp.Generator(channel_count=2).setup({}, {})[PORT_OUT]
    print(f"Generator port: {port[KEYS.CHANNEL_COUNT]} channels, "
          f"{port[KEYS.CHANNEL_ROLES]}")

    # 4. The sink says what it needs, and refuses without it.
    try:
        SqliteWriter(file_name=DATABASE).setup({}, {PORT_IN: {}})
    except ValueError as error:
        print(f"refused: {error}")
