"""Remove baseline drift with a 0.5 Hz highpass.

The generator's 0.01 Hz sine stands in for the slow drift a real
electrode produces over minutes. The filter leaves the faster noise and
recenters the trace on zero.

Requires: pip install gpype[gui]
Run: python example_basic_highpass.py
"""
import gpype as gp


if __name__ == "__main__":
    # Create the main application window
    app = gp.MainApp()

    # Create processing pipeline
    p = gp.Pipeline()

    # Generate signal with slow baseline drift + noise
    source = gp.Generator(
        sampling_rate=250,
        channel_count=8,
        signal_amplitude=50,  # Large slow oscillation
        signal_frequency=0.01,  # Very slow (60 sec period)
        noise_amplitude=5,
    )  # Higher frequency noise

    # Highpass filter to remove baseline drift
    filter = gp.Highpass(f_c=0.5)  # 0.5 Hz cutoff (removes < 0.5 Hz)

    # Real-time visualization scope
    scope = gp.TimeSeriesScope(
        amplitude_limit=30, time_window=10  # Y-axis range
    )  # 10 seconds display

    # Connect processing chain: drift signal -> highpass -> display
    p.connect(source, filter)
    p.connect(filter, scope)

    # Add scope to application window
    app.add_widget(scope)

    # Start baseline drift removal demonstration
    p.start()  # Begin signal processing
    app.run()  # Show GUI and start main loop
    p.stop()  # Clean shutdown when window closes
