"""Season 7, Episode 4: a control surface, read and set while it runs."""

import threading
import time

import numpy as np

import gpype as gp

PORT_IN = gp.Constants.Defaults.PORT_IN
PORT_OUT = gp.Constants.Defaults.PORT_OUT


class Gain(gp.IONode):
    """Multiplies the signal by a gain an operator can move."""

    class Configuration(gp.IONode.Configuration):
        class Keys(gp.IONode.Configuration.Keys):
            GAIN = "gain"

    def __init__(self, gain: float = 1.0, **kwargs):
        super().__init__(gain=gain, **kwargs)
        self._gain = float(gain)
        self._peak = 0.0
        # The control surface is called from the caller's thread, step()
        # from the source's.
        self._lock = threading.Lock()

    def step(self, data):
        out = data[PORT_IN] * self._gain
        with self._lock:
            self._peak = max(self._peak, float(np.abs(out).max()))
        return {PORT_OUT: out}

    @gp.controllable
    def get_control(self) -> dict:
        """The values a caller may read and set."""
        return {"gain": self._gain}

    @gp.controllable
    def set_control(self, arg: dict) -> dict:
        """Handed every key of get_control(), changed or not."""
        gain = float(arg["gain"])
        if gain < 0:
            raise ValueError(f"gain must not be negative, got {gain}")
        self._gain = gain
        return self.get_control()

    @gp.action
    def take_peak(self, arg: dict) -> dict:
        """Report the largest output since the last call, and reset."""
        with self._lock:
            peak, self._peak = self._peak, 0.0
        return {"peak": round(peak, 1)}


if __name__ == "__main__":

    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=250, channel_count=2,
                          signal_frequency=10, signal_amplitude=10.0)
    gain = Gain(gain=1.0, name="gain")
    p.connect(source, gain)
    p.connect(gain, gp.Collector())

    # What a controlling process sees: every node, its id and surface.
    entry = next(n for n in p.get_nodes() if n["class"] == "Gain")
    node = entry["id"]
    print(f"controllable: {entry['controllable']}, "
          f"actions: {entry['actions']}")

    p.start()
    time.sleep(1.0)
    print(f"before: {p.get_control(node)}, "
          f"{p.invoke_action(node, 'take_peak', {})}")

    # A set names only what changes, and returns the whole state.
    print(f"set:    {p.set_control(node, {'gain': 2.0})}")
    time.sleep(1.0)
    print(f"after:  {p.get_control(node)}, "
          f"{p.invoke_action(node, 'take_peak', {})}")

    # A key get_control() does not report is refused before the node
    # sees it.
    try:
        p.set_control(node, {"gian": 3.0})
    except KeyError as error:
        print(f"refused: {error.args[0]}")

    # A control is not configuration: the saved pipeline keeps the gain
    # it was built with.
    saved = next(n for n in p.serialize()["nodes"] if n["class"] == "Gain")
    print(f"saved:  gain={saved['config']['gain']}")
    p.stop()
    p.close()
