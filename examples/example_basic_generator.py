"""
Basic Generator Example - Synthetic Signal Generation and Visualization

This is the simplest g.Pype example, demonstrating the fundamental concepts
of signal generation and real-time visualization. It serves as an introduction
to the g.Pype framework and shows the minimal pipeline needed for signal
processing applications.

What this example shows:
- Creating synthetic EEG-like signals using the Generator node
- Real-time visualization with TimeSeriesScope
- Basic two-node pipeline construction: Generator -> Scope
- Essential g.Pype concepts: nodes, connections, and widgets

Expected output:
When you run this example, you'll see:
- 8 channels of clean 10 Hz sine waves with slight background noise
- Real-time display showing 10 seconds of data history
- Smooth signal visualization at 250 Hz sampling rate

This example is intended for:
- First-time g.Pype users learning the basics
- Testing g.Pype installation and GUI functionality
- Understanding signal parameters and their visual effects
- Learning the fundamental pipeline construction pattern

Signal parameters:
- Sampling rate: 250 Hz (common for EEG applications)
- Channels: 8 (typical EEG montage)
- Frequency: 10 Hz (alpha rhythm simulation)
- Amplitude: 10 µV (clearly visible)
- Shape: Sine wave (clean, predictable waveform)
- Noise: Minimal (1 unit) for realistic signal appearance

Usage:
    python example_basic_generator.py

Note:
    This is the foundation example - all other examples build upon these
    basic concepts of signal generation and visualization.
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
