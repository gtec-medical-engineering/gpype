# --------------------------------------------------------------
# Example file s1e2_hello_world.py
# For details and usage, see g.Pype Training Season 1, Episode 2
# --------------------------------------------------------------

import gpype as gp

if __name__ == "__main__":

    # Create main app and pipeline
    app = gp.MainApp()
    p = gp.Pipeline()

    # Create signal source and scope
    source = gp.Generator(signal_amplitude=25)
    scope = gp.TimeSeriesScope()

    # Connect nodes
    p.connect(source, scope)

    # Add widget to main app
    app.add_widget(scope)

    # Start pipeline and run application
    p.start()
    app.run()  # blocking until window is closed
    p.stop()
