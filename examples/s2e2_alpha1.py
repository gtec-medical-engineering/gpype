# --------------------------------------------------------------
# Example file s2e2_alpha1.py
# For details and usage, see g.Pype Training Season 2, Episode 2
# --------------------------------------------------------------

import gpype as gp

if __name__ == "__main__":

    # Create main app and pipeline
    app = gp.MainApp()
    p = gp.Pipeline()

    # Create alpha source
    source_alpha = gp.Generator(channel_count=1,
                                signal_frequency=10,
                                signal_amplitude=25)

    source_mod = gp.Generator(channel_count=1,
                              signal_frequency=0.2,
                              signal_amplitude=25)

    source_noise = gp.Generator(channel_count=1,
                                noise_amplitude=25)

    # Create router to combine signals
    router = gp.Router(input_channels={"alpha": [0],
                                       "modulator": [0],
                                       "noise": [0]})

    # Create scope
    scope = gp.TimeSeriesScope()

    # Connect nodes
    p.connect(source_alpha, router["alpha"])
    p.connect(source_mod, router["modulator"])
    p.connect(source_noise, router["noise"])
    p.connect(router, scope)

    # Add widget to main app
    app.add_widget(scope)

    # Start pipeline and run application
    p.start()
    app.run()  # blocking until window is closed
    p.stop()
