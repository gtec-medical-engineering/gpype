import os

import gpype as gp

parent_dir = os.path.dirname(os.path.abspath(__file__))
paradigm_root_folder = os.path.join(parent_dir, "paradigms")
fs = 250

if __name__ == "__main__":

    # main app
    app = gp.MainApp()

    # pipeline
    p = gp.Pipeline()

    # signal generator
    source = gp.Generator(
        sampling_rate=fs,
        channel_count=8,
        signal_frequency=10,
        signal_amplitude=10,
        signal_shape="sine",
        noise_amplitude=10,
    )

    # presenter trigger via UDP
    trigger = gp.UDPReceiver()

    # router
    router = gp.Router(input_selector=[gp.Router.ALL, gp.Router.ALL])

    # scope
    mk = gp.TimeSeriesScope.Markers
    markers = [
        mk(color="r", label="task 1", channel=8, value=1),
        mk(color="b", label="task 2", channel=8, value=2),
        mk(color="k", label="task 3", channel=8, value=3),
    ]
    scope = gp.TimeSeriesScope(
        amplitude_limit=30, time_window=10, markers=markers
    )

    # connect nodes
    p.connect(source, router["in1"])
    p.connect(trigger, router["in2"])
    p.connect(router, scope)

    # presenter
    presenter = gp.ParadigmPresenter(paradigm_root_folder)

    # add widgets
    app.add_widget(presenter)
    app.add_widget(scope)

    # start pipeline and main app
    p.start()
    app.run()
    p.stop()
