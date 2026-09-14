"""Writing your own node: one class, three things to fill in.

DriftMonitor reports how far each channel has drifted from its own running
mean -- the first sign of an electrode losing contact, and nothing built in
reports it. Declare parameters in a ``Configuration.Keys`` holder or they
are dropped when the pipeline is serialised; negotiate the output context
in ``setup()``, the first place the sampling rate is known; do the work in
``step()``.

Requires: the gui extra (TimeSeriesScope)
Run: python example_custom_node.py
"""

import numpy as np

import gpype as gp

PORT_IN = gp.Constants.Defaults.PORT_IN
PORT_OUT = gp.Constants.Defaults.PORT_OUT

SAMPLING_RATE = 250
CHANNELS = 4


class DriftMonitor(gp.IONode):
    """Reports each channel's departure from its own running mean.

    One output channel per input channel, same rate, same frame size --
    so this is a node that changes the *meaning* of the numbers without
    changing the shape of the stream.
    """

    class Configuration(gp.IONode.Configuration):
        """Parameters, declared so they survive serialisation."""

        class Keys(gp.IONode.Configuration.Keys):
            """Keys for DriftMonitor."""

            #: Seconds of history the running mean is taken over.
            WINDOW_SECONDS = "window_seconds"

    def __init__(self, window_seconds: float = 2.0, **kwargs):
        """Initialize the monitor.

        Args:
            window_seconds: Seconds of history the running mean covers.
                Longer reacts more slowly and is less fooled by a real
                signal; shorter notices a lost electrode sooner.
            **kwargs: Additional arguments for IONode.

        Raises:
            ValueError: If the window is not positive.
        """
        if window_seconds <= 0:
            raise ValueError(
                f"window_seconds must be positive, got {window_seconds}"
            )
        super().__init__(window_seconds=window_seconds, **kwargs)
        self._mean = None
        self._decay = None

    def setup(self, data: dict, port_metadata_in: dict) -> dict:
        """Size the running mean from the rate the input actually has.

        The window is given in seconds, and a frame is given in samples,
        so the conversion needs the sampling rate -- which is knowable
        here and nowhere earlier.

        Args:
            data: The first input frames.
            port_metadata_in: Context of every input port.

        Returns:
            dict: Context of every output port.
        """
        context = port_metadata_in[PORT_IN]
        rate = context[gp.Constants.Keys.SAMPLING_RATE]
        channels = context[gp.Constants.Keys.CHANNEL_COUNT]
        window = self.config[self.Configuration.Keys.WINDOW_SECONDS]

        # One-pole smoother: the weight a single sample carries.
        self._decay = 1.0 / max(1.0, float(rate) * float(window))
        self._mean = np.zeros((1, channels), dtype=gp.Constants.DATA_TYPE)

        # Same rate, same width, same frame size: only the meaning
        # changes. Returning the input context unchanged says exactly
        # that, and says it in one line.
        return {PORT_OUT: dict(context)}

    def step(self, data: dict) -> dict:
        """Return each sample's distance from its channel's mean.

        Args:
            data: Input frames by port name.

        Returns:
            dict: Output frames by port name.
        """
        frame = data[PORT_IN]
        out = np.empty_like(frame)
        for index in range(frame.shape[0]):
            row = frame[index : index + 1, :]
            self._mean += self._decay * (row - self._mean)
            out[index, :] = row - self._mean
        return {PORT_OUT: out}


if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    source = gp.Generator(
        sampling_rate=SAMPLING_RATE,
        channel_count=CHANNELS,
        signal_frequency=10,
        signal_amplitude=20.0,
        noise_amplitude=2.0,
        frame_size=25,
    )
    drift = DriftMonitor(window_seconds=2.0)
    scope = gp.TimeSeriesScope(amplitude_limit=50, time_window=10)

    p.connect(source, drift)
    p.connect(drift, scope)
    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()
