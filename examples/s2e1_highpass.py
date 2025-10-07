# --------------------------------------------------------------
# Example file s2e1_a_highpass.py
# For details and usage, see g.Pype Training Season 2, Episode 1
# --------------------------------------------------------------

import gpype as gp

if __name__ == "__main__":

    # Create main app and pipeline
    app = gp.MainApp()
    p = gp.Pipeline()

    # Create signal source
    source = gp.Generator(signal_frequency=0.05,
                          signal_amplitude=75,
                          noise_amplitude=15)

    # Create highpass filter
    highpass = gp.Highpass(f_c=1)

    # Create scope
    scope = gp.TimeSeriesScope()

    # Connect nodes
    p.connect(source, highpass)
    p.connect(highpass, scope)

    # Add widget to main app
    app.add_widget(scope)

    # Start pipeline and run application
    p.start()
    app.run()  # blocking until window is closed
    p.stop()
