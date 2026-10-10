"""Season 3, Episode 6: key presses as an event channel beside the signal."""

import gpype as gp

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    source = gp.Generator(sampling_rate=250, channel_count=8,
                          signal_frequency=10, signal_amplitude=10,
                          noise_amplitude=10)

    # Every press becomes one sample carrying the key's code
    keyboard = gp.Keyboard()

    # The generator's 8 channels, then the keyboard's one: the key codes
    # land on channel 8, counting from 0
    router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])

    # A marker per key, pinned to that channel; the values are key codes
    mk = gp.TimeSeriesScope.Markers
    markers = [mk(color="r", label="up", channel=8, value=38),
               mk(color="g", label="right", channel=8, value=39),
               mk(color="c", label="down", channel=8, value=40),
               mk(color="y", label="left", channel=8, value=37)]

    # The code channel itself is hidden: a raw 40 would overrun the lane
    scope = gp.TimeSeriesScope(amplitude_limit=30, time_window=10,
                               markers=markers, hidden_channels=[8])

    p.connect(source, router["in1"])
    p.connect(keyboard, router["in2"])
    p.connect(router, scope)
    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()
