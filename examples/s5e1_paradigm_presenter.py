"""Season 5, Episode 1: a paradigm's stimulus codes, received as they fire.

Windows only: the widget drives g.tec's Paradigm Presenter, installed
and licensed separately; its Python package comes with gpype[devices].
"""

import os

import gpype as gp

# Every paradigm in this folder sends its codes to 127.0.0.1, port 1000
PARADIGMS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "paradigms")

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    source = gp.Generator(sampling_rate=250, channel_count=8,
                          signal_frequency=10, signal_amplitude=10,
                          noise_amplitude=10)

    # The presenter's codes arrive here: port 1000 is the default on
    # both ends
    receiver = gp.UDPReceiver()

    # The generator's 8 channels, then the codes on channel 8
    router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])

    mk = gp.TimeSeriesScope.Markers
    markers = [mk(color="r", label="task 1", channel=8, value=1),
               mk(color="c", label="task 2", channel=8, value=2),
               mk(color="y", label="task 3", channel=8, value=3)]
    scope = gp.TimeSeriesScope(amplitude_limit=30, time_window=10,
                               markers=markers, hidden_channels=[8])

    p.connect(source, router["in1"])
    p.connect(receiver, router["in2"])
    p.connect(router, scope)

    # The widget is not connected to the pipeline: it lists the folder's
    # paradigms, and starts and stops the one you pick
    presenter = gp.ParadigmPresenter(PARADIGMS)
    app.add_widget(presenter)
    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()
