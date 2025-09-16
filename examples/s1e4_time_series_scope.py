# --------------------------------------------------------------
# Example file s1e4_time_series_scope.py
# For details and usage, see g.Pype Training Season 1, Episode 4
# --------------------------------------------------------------

import gpype as gp

if __name__ == "__main__":

    # Create main app and pipeline
    app = gp.MainApp()
    p = gp.Pipeline()

    # Create signal source
    source = gp.Generator(signal_amplitude=25)

    # Create scope
    scope = gp.TimeSeriesScope(time_window=10,
                               amplitude_limit=50,
                               hidden_channels=[])

    # Connect nodes
    p.connect(source, scope)

    # Add widget to main app
    app.add_widget(scope)

    # Start pipeline and run application
    p.start()
    app.run()  # blocking until window is closed
    p.stop()
