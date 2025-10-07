# --------------------------------------------------------------
# Example file s2e1_d_bandstop.py
# For details and usage, see g.Pype Training Season 2, Episode 1
# --------------------------------------------------------------

import gpype as gp

if __name__ == "__main__":

    # Create main app and pipeline
    app = gp.MainApp()
    p = gp.Pipeline()

    # Create signal source
    source = gp.Generator(signal_frequency=50,
                          signal_amplitude=75,
                          noise_amplitude=15)

    # Create bandstop filter
    bandstop = gp.Bandstop(f_lo=48, f_hi=52)

    # Create scope
    scope = gp.TimeSeriesScope()

    # Connect nodes
    p.connect(source, bandstop)
    p.connect(bandstop, scope)

    # Add widget to main app
    app.add_widget(scope)

    # Start pipeline and run application
    p.start()
    app.run()  # blocking until window is closed
    p.stop()
