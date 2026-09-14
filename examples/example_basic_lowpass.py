"""Lowpass filtering broadband noise, shown live.

A 5 Hz cutoff at 250 Hz sampling: what is left is a slow, smooth
version of the noise that went in.

Requires: gpype[gui]
Run: python example_basic_lowpass.py
"""
import gpype as gp


if __name__ == "__main__":

    # Create the main application window
    app = gp.MainApp()

    # Create processing pipeline
    p = gp.Pipeline()

    # Generate broadband noise (all frequencies present)
    source = gp.Generator(
        sampling_rate=250,  # 250 Hz sampling rate
        channel_count=8,  # 8 channels
        noise_amplitude=30,
    )  # Pure noise

    # Lowpass filter to remove high-frequency components
    f_c = 5  # Cutoff frequency in Hz
    filter = gp.Lowpass(f_c=f_c)  # Remove frequencies above 5 Hz

    # Real-time visualization scope
    scope = gp.TimeSeriesScope(
        amplitude_limit=30, time_window=10  # Y-axis range
    )  # 10 seconds display

    # Connect processing chain: noise -> lowpass filter -> display
    p.connect(source, filter)
    p.connect(filter, scope)

    # Add scope to application window
    app.add_widget(scope)

    # Start filtering and visualization
    p.start()  # Begin signal processing (watch noise smoothing)
    app.run()  # Show GUI and start main loop
    p.stop()  # Clean shutdown when window closes
