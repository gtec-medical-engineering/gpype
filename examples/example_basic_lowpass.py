"""
Basic Lowpass Filter Example - High-Frequency Noise Reduction

This example demonstrates how to use a lowpass filter to remove high-frequency
noise and artifacts from signals while preserving low-frequency content. This
is fundamental for cleaning noisy signals and extracting slow brain rhythms
commonly analyzed in BCI applications.

What this example shows:
- Generating broadband noise (high-frequency content across spectrum)
- Applying a 5 Hz lowpass filter to remove high-frequency components
- Real-time visualization of the smoothing effect
- Basic pipeline construction: Generator -> Lowpass -> Scope

Expected output:
When you run this example, you'll see:
- Dramatically smoothed signals with high-frequency noise removed
- Much cleaner, slower-varying waveforms compared to input noise
- Signals that appear more like typical low-frequency EEG rhythms
- Clear demonstration of how lowpass filters create smoother signals

Real-world applications:
- EEG slow cortical potential analysis (< 1 Hz)
- Delta rhythm extraction (0.5-4 Hz) for sleep studies
- Removing muscle artifacts (EMG) from EEG (typically > 30 Hz)
- Anti-aliasing before downsampling signals
- Smoothing signals for trend analysis

Technical details:
- Cutoff frequency: 5 Hz (removes frequencies above 5 Hz)
- Filter type: Lowpass (attenuates high frequencies, passes low frequencies)
- Input signal: Pure broadband noise (all frequencies present)
- Effect: High-frequency content suppressed, low-frequency content preserved

Common BCI use cases:
- Slow cortical potential (SCP) detection
- Removing high-frequency artifacts before feature extraction
- Delta/theta rhythm analysis for cognitive state assessment
- Preprocessing for low-frequency BCI paradigms
- Signal conditioning for slow event-related potentials

Usage:
    python example_basic_lowpass.py

Note:
    Compare with example_basic_highpass.py to see complementary filtering:
    - Lowpass: removes high frequencies, keeps low frequencies
    - Highpass: removes low frequencies, keeps high frequencies
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
