# --------------------------------------------------------------
# Example file s2e2_mixer.py
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

    # Create router to split signals
    splitter = gp.Router(output_channels=[[0, 3], [1, 2]])

    # Create router to mix signals
    mixer = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL],
                      output_channels=[[0, 2], [3, 1]])

    # Create scopes
    scope1 = gp.TimeSeriesScope()
    scope2 = gp.TimeSeriesScope()

    # Connect nodes
    p.connect(source1, combiner["in1"])
    p.connect(source2, combiner["in2"])
    p.connect(source3, combiner["in3"])
    p.connect(source4, combiner["in4"])
    p.connect(combiner, splitter)
    p.connect(splitter["out1"], mixer["in1"])
    p.connect(splitter["out2"], mixer["in2"])
    p.connect(mixer["out1"], scope1)
    p.connect(mixer["out2"], scope2)

    # Add widget to main app
    app.add_widget(scope1)
    app.add_widget(scope2)

    # Start pipeline and run application
    p.start()
    app.run()  # blocking until window is closed
    p.stop()
