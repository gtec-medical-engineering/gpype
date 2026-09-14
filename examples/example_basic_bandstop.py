"""Remove 50 Hz power line interference with a 48-52 Hz bandstop.

50 Hz is the European mains frequency; where the mains runs at 60 Hz,
use 58-62 Hz instead. This is the complement of example_basic_bandpass.py:
a bandstop keeps everything except the stated band.

Requires: pip install gpype[gui]
Run: python example_basic_bandstop.py
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
        channel_count=8,
        signal_frequency=50,
        signal_amplitude=100,
        signal_shape="sine",
        noise_amplitude=10,
    )

    # bandstop
    filter = gp.Bandstop(f_lo=48, f_hi=52)

    # scope
    scope = gp.TimeSeriesScope(amplitude_limit=30, time_window=10)

    # connect nodes
    p.connect(source, filter)
    p.connect(filter, scope)

    # add widgets
    app.add_widget(scope)

    # start pipeline and main app
    p.start()
    app.run()
    p.stop()
