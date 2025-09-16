# --------------------------------------------------------------
# Example file s2e1_b_lowpass.py
# For details and usage, see g.Pype Training Season 2, Episode 1
# --------------------------------------------------------------

import gpype as gp

if __name__ == "__main__":

    # Create main app and pipeline
    app = gp.MainApp()
    p = gp.Pipeline()

    # Create signal source
    source = gp.Generator(signal_frequency=4,
                          signal_amplitude=25,
                          noise_amplitude=25)

    # Create lowpass filter
    lowpass = gp.Lowpass(f_c=6)

    # Create scope
    scope = gp.TimeSeriesScope()

    # Connect nodes
    p.connect(source, lowpass)
    p.connect(lowpass, scope)

    # Add widget to main app
    app.add_widget(scope)

    # Start pipeline and run application
    p.start()
    app.run()  # blocking until window is closed
    p.stop()
