"""Delay one channel by 125 samples (0.5 s at 250 Hz) and plot both.

A Router splits the generator's two channels into separate paths, one of
them through the Delay, and a second Router merges them for display.
Both carry the same 1 Hz sine, so half a period of delay puts the traces
in antiphase. The noise is drawn per channel and does not match.

Requires: pip install gpype[gui]
Run: python example_basic_delay.py
"""
import gpype as gp


if __name__ == "__main__":

    # main app
    app = gp.MainApp()

    # pipeline
    p = gp.Pipeline()

    # signal generator
    source = gp.Generator(
        sampling_rate=250,
        channel_count=2,
        signal_frequency=1,
        signal_amplitude=10,
        signal_shape="sine",
        noise_amplitude=1,
    )

    # split signals
    splitter = gp.Router(
        input_channels=[gp.Router.ALL], output_channels=[[0], [1]]
    )

    # delay one signal by 125 samples (0.5 seconds at 250 Hz)
    delay = gp.Delay(num_samples=125)

    # merge signals back together
    merger = gp.Router(
        input_channels=[[0], [0]], output_channels=[gp.Router.ALL]
    )

    # scope
    scope = gp.TimeSeriesScope(amplitude_limit=30, time_window=10)

    # connect nodes
    p.connect(source, splitter)
    p.connect(splitter["out1"], merger["in1"])
    p.connect(splitter["out2"], delay)
    p.connect(delay, merger["in2"])
    p.connect(merger, scope)

    # add widgets
    app.add_widget(scope)

    # start pipeline and main app
    p.start()
    app.run()
    p.stop()
