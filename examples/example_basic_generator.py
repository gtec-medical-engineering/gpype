"""The smallest g.Pype pipeline: a Generator straight into a scope.

Start here -- every other example adds nodes between these two.

Requires: pip install gpype[gui]
Run: python example_basic_generator.py
"""
import gpype as gp


if __name__ == "__main__":
    # Create the main application window
    app = gp.MainApp()

    # Create a processing pipeline to connect nodes
    p = gp.Pipeline()

    # Generate synthetic 8-channel EEG-like signals
    source = gp.Generator(
        sampling_rate=250,  # 250 Hz sampling
        channel_count=8,  # 8 parallel channels
        signal_frequency=10,  # 10 Hz alpha rhythm
        signal_amplitude=10,  # Clear signal strength
        signal_shape="sine",  # Clean sine waves
        noise_amplitude=1,
    )  # Minimal background noise

    # Real-time visualization scope
    scope = gp.TimeSeriesScope(
        amplitude_limit=30, time_window=10  # Y-axis: ±30 units
    )  # X-axis: 10 seconds

    # Connect generator directly to scope (simplest pipeline)
    p.connect(source, scope)

    # Add scope widget to the application window
    app.add_widget(scope)

    # Start pipeline and run application
    p.start()  # Begin signal generation and display
    app.run()  # Show GUI and start main loop
    p.stop()  # Clean shutdown when window closes
