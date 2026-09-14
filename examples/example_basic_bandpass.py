"""Isolate a 10 Hz sine from noise with a 9-11 Hz bandpass.

The generator's noise is broadband, so the scope shows the filter doing
the work: a clean 10 Hz wave where the unfiltered signal looks ragged.

Requires: pip install gpype[gui]
Run: python example_basic_bandpass.py
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
        signal_frequency=10,
        signal_amplitude=10,
        signal_shape="sine",
        noise_amplitude=1,
        frame_size=10,
    )

    # bandpass
    filter = gp.Bandpass(f_lo=9, f_hi=11)

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
