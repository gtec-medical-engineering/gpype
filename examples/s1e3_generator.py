# --------------------------------------------------------------
# Example file s1e3_generator.py
# For details and usage, see g.Pype Training Season 1, Episode 3
# --------------------------------------------------------------

import gpype as gp

if __name__ == "__main__":

    # Create main app and pipeline
    app = gp.MainApp()
    p = gp.Pipeline()

    # Create signal source
    source = gp.Generator(channel_count=8,
                          signal_frequency=10,
                          signal_shape=gp.Generator.SHAPE_SINUSOID,
                          signal_amplitude=25,
                          noise_amplitude=0)

    # Create scope
    scope = gp.TimeSeriesScope()

    # Connect nodes
    p.connect(source, scope)

    # Add widget to main app
    app.add_widget(scope)

    # Start pipeline and run application
    p.start()
    app.run()  # blocking until window is closed
    p.stop()
