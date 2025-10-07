# --------------------------------------------------------------
# Example file s2e2_combiner.py
# For details and usage, see g.Pype Training Season 2, Episode 2
# --------------------------------------------------------------

import gpype as gp

if __name__ == "__main__":

    # Create main app and pipeline
    app = gp.MainApp()
    p = gp.Pipeline()

    # Create 4 sources with different frequencies
    source1, source2, source3, source4 = [
        gp.Generator(channel_count=1,
                     signal_frequency=f,
                     signal_amplitude=25) for f in (2, 4, 6, 8)
    ]

    # Create router to combine signals
    combiner = gp.Router(input_channels=[[0], [0], [0], [0]])

    # Create scope
    scope = gp.TimeSeriesScope()

    # Connect nodes
    p.connect(source1, combiner["in1"])
    p.connect(source2, combiner["in2"])
    p.connect(source3, combiner["in3"])
    p.connect(source4, combiner["in4"])
    p.connect(combiner, scope)

    # Add widget to main app
    app.add_widget(scope)

    # Start pipeline and run application
    p.start()
    app.run()  # blocking until window is closed
    p.stop()
