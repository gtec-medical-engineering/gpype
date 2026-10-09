"""Season 7, Episode 2: named ports, checked at start, and an event output."""

import time

import numpy as np

import gpype as gp

KEYS = gp.Constants.Keys
TIMING = gp.Constants.Timing

RATE = 250
SECONDS = 5


class Crossings(gp.IONode):
    """Signal minus reference, and an event at each rising crossing."""

    PORT_SIGNAL = "signal"
    PORT_REFERENCE = "reference"
    PORT_OUT = gp.Constants.Defaults.PORT_OUT
    PORT_CROSSING = "crossing"

    class Configuration(gp.IONode.Configuration):
        class Keys(gp.IONode.Configuration.Keys):
            LEVEL = "level"

    def __init__(self, level: float = 5.0, **kwargs):
        # Taken from kwargs when present: a node loaded from a saved
        # pipeline is handed its ports back there.
        input_ports = kwargs.pop(KEYS.INPUT_PORTS, [
            gp.IPort.Configuration(name=self.PORT_SIGNAL),
            gp.IPort.Configuration(name=self.PORT_REFERENCE),
        ])
        output_ports = kwargs.pop(KEYS.OUTPUT_PORTS, [
            gp.OPort.Configuration(name=self.PORT_OUT),
            gp.OPort.Configuration(name=self.PORT_CROSSING,
                                   timing=TIMING.ASYNC),
        ])
        super().__init__(level=level, input_ports=input_ports,
                         output_ports=output_ports, **kwargs)
        self._above = False

    def setup(self, data, port_context_in):
        # The base class refuses inputs it cannot combine, and describes
        # every output port like the combined input.
        port_context_out = super().setup(data, port_context_in)
        # An event is one value, arriving whenever it happens.
        port_context_out[self.PORT_CROSSING].update({
            KEYS.CHANNEL_COUNT: 1,
            KEYS.FRAME_SIZE: 1,
            KEYS.CHANNEL_ROLES: [gp.Constants.ChannelRoles.SIGNAL],
            KEYS.CHANNEL_LABELS: ["crossing"],
            gp.IPort.Configuration.Keys.TIMING: TIMING.ASYNC,
        })
        self._above = False
        return port_context_out

    def step(self, data):
        difference = data[self.PORT_SIGNAL] - data[self.PORT_REFERENCE]
        mean = difference.mean(axis=1)
        above = mean > self.config[self.Configuration.Keys.LEVEL]
        before = np.concatenate(([self._above], above[:-1]))
        self._above = bool(above[-1])

        out = {self.PORT_OUT: difference}
        # Most steps put nothing on the asynchronous port.
        rising = np.flatnonzero(above & ~before)
        if rising.size:
            out[self.PORT_CROSSING] = mean[rising[:1], None]
        return out


def build(reference_channels: int) -> tuple:
    """A 4-channel square wave against a reference of noise."""
    p = gp.Pipeline()
    signal = gp.Generator(sampling_rate=RATE, channel_count=4,
                          signal_frequency=0.5, signal_shape="rect",
                          signal_amplitude=10.0, noise_amplitude=1.0,
                          name="signal")
    reference = gp.Generator(sampling_rate=RATE,
                             channel_count=reference_channels,
                             noise_amplitude=1.0, name="reference")
    node = Crossings(level=5.0)
    difference = gp.Collector(name="difference")
    events = gp.Collector(name="events")

    p.connect(signal, node[Crossings.PORT_SIGNAL])
    p.connect(reference, node[Crossings.PORT_REFERENCE])
    p.connect(node, difference)
    p.connect(node[Crossings.PORT_CROSSING], events)
    return p, difference, events


if __name__ == "__main__":

    # 1. Four channels against two. connect() accepts it, because the
    #    widths are known only once the sources have set up.
    p, _, _ = build(reference_channels=2)
    try:
        p.start()
    except RuntimeError as error:
        print(f"refused: {error.__cause__}")
    finally:
        p.stop()
        p.close()

    # 2. A one-channel reference is subtracted from all four.
    p, difference, events = build(reference_channels=1)
    p.start()
    while (difference.result is None
           or len(difference.result.data) < RATE * SECONDS):
        time.sleep(0.05)
    p.stop()
    p.close()

    # The square wave rises at 0, 2 and 4 s. A Collector stacks the
    # events into (time, channel, event).
    print(f"{SECONDS} s of difference: "
          f"{difference.result.data.shape[1]} channels")
    print(f"events: {events.result.data.shape}")
