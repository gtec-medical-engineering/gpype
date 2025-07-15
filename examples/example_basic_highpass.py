"""
Basic Highpass Filter Example - Baseline Drift Removal

This example demonstrates how to use a highpass filter to remove very low-
frequency components (baseline drift) from signals while preserving higher
frequency content. This is essential for removing DC offsets and slow drifts
commonly found in EEG and BCI recordings.

What this example shows:
- Generating a very slow 0.01 Hz sine wave (simulating baseline drift)
- Adding higher frequency noise components
- Applying a 0.5 Hz highpass filter to remove the slow drift
- Real-time visualization showing drift removal effect

Expected output:
When you run this example, you'll see:
- After filtering: Drift removed, leaving only the faster noise components
- Stable baseline with centered signal around zero
- Clear demonstration of how highpass filters eliminate slow variations

Real-world applications:
- EEG baseline drift correction (electrode movement, skin conductance changes)
- DC offset removal from amplifier outputs
- Skin potential artifact elimination
- Preparation for AC-coupled analysis

Usage:
    python example_basic_highpass.py

Note:
    The very slow 0.01 Hz signal simulates real baseline drift that occurs
    over minutes in actual EEG recordings due to electrode impedance changes.
"""


import gpype as gp


if __name__ == '__main__':
    # Create the main application window
    app = gp.MainApp()

    # Create processing pipeline
    p = gp.Pipeline()

    # Generate signal with slow baseline drift + noise
    source = gp.Generator(sampling_rate=250,
                          channel_count=8,
                          signal_amplitude=50,      # Large slow oscillation
                          signal_frequency=0.01,    # Very slow (60 sec period)
                          noise_amplitude=5)        # Higher frequency noise

    # Highpass filter to remove baseline drift
    filter = gp.Highpass(f_c=0.5)  # 0.5 Hz cutoff (removes < 0.5 Hz)

    # Real-time visualization scope
    scope = gp.TimeSeriesScope(amplitude_limit=30,  # Y-axis range
                               time_window=10)       # 10 seconds display

    # Connect processing chain: drift signal -> highpass -> display
    p.connect(source, filter)
    p.connect(filter, scope)

    # Add scope to application window
    app.add_widget(scope)

    # Start baseline drift removal demonstration
    p.start()      # Begin signal processing
    app.run()      # Show GUI and start main loop
    p.stop()       # Clean shutdown when window closes
