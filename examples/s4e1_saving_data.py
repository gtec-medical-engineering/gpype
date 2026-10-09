"""Season 4, Episode 1: signal and key presses recorded to a CSV file."""

import gpype as gp

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    source = gp.Generator(sampling_rate=250, channel_count=8,
                          signal_frequency=10, signal_amplitude=10,
                          noise_amplitude=10)
    keyboard = gp.Keyboard()
    router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])

    mk = gp.TimeSeriesScope.Markers
    markers = [mk(color="r", label="up", channel=8, value=38),
               mk(color="g", label="right", channel=8, value=39),
               mk(color="c", label="down", channel=8, value=40),
               mk(color="y", label="left", channel=8, value=37)]
    scope = gp.TimeSeriesScope(amplitude_limit=30, time_window=10,
                               markers=markers, hidden_channels=[8])

    # The writer adds the date and time to the name, so every run
    # writes a new file into the folder the script runs from
    writer = gp.CsvWriter(file_name="s4e1.csv")

    p.connect(source, router["in1"])
    p.connect(keyboard, router["in2"])
    p.connect(router, scope)
    p.connect(router, writer)  # the file holds what the scope shows
    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()
