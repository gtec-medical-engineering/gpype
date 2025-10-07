# --------------------------------------------------------------
# Example file s2e1_c_bandpass.py
# For details and usage, see g.Pype Training Season 2, Episode 1
# --------------------------------------------------------------

import gpype as gp

if __name__ == "__main__":

    # Create main app and pipeline
    app = gp.MainApp()
    p = gp.Pipeline()

    # Create signal source
    source = gp.Generator(signal_frequency=10,
                          signal_amplitude=25,
                          noise_amplitude=25)

    # Create bandpass filter
    bandpass = gp.Bandpass(f_lo=8, f_hi=12)

    # Create scope
    scope = gp.TimeSeriesScope()

    # Connect nodes
    p.connect(source, bandpass)
    p.connect(bandpass, scope)

    # Add widget to main app
    app.add_widget(scope)

    # Start pipeline and run application
    p.start()
    app.run()  # blocking until window is closed
    p.stop()
