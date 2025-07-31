"""
Basic FFT (Fast Fourier Transform) Example - Frequency Domain Analysis

This example demonstrates how to perform frequency domain analysis using FFT
to convert time-domain signals into their frequency components. This is
fundamental for BCI applications that analyze spectral content of neural
signals.

What this example shows:
- Generating a 10 Hz rectangular wave signal with noise
- Applying FFT with windowing to convert to frequency domain
- Real-time spectrum visualization showing frequency peaks
- Pipeline: Generator -> FFT -> Spectrum Scope

Expected output:
The spectrum scope will display:
- Strong peak at 10 Hz (fundamental frequency)
- Additional peaks at 30, 50, 70 Hz... (odd harmonics of rectangular wave)
- Background noise floor across all frequencies

This demonstrates how rectangular waves contain multiple frequency components
(harmonics), unlike pure sine waves which show only a single peak.

Real-world applications:
- EEG rhythm analysis (alpha, beta, gamma bands)
- Motor imagery classification (mu/beta suppression)
- SSVEP detection (steady-state visual evoked potentials)
- Artifact identification in frequency domain
- Power spectral density analysis

Technical details:
- Window size: 250 samples (1 second at 250 Hz)
- Overlap: 50% (smooth spectral updates)
- Window function: Hamming (reduces spectral leakage)
- Frequency resolution: 1 Hz (250 Hz / 250 samples)

Usage:
    python example_basic_fft.py
"""
import gpype as gp

fs = 250  # Sampling frequency in Hz

if __name__ == "__main__":

    # Create the main application window
    app = gp.MainApp()

    # Create processing pipeline
    p = gp.Pipeline()

    # Generate 10 Hz rectangular wave (rich in harmonics)
    source = gp.Generator(
        sampling_rate=fs,
        channel_count=2,  # Dual channel
        signal_frequency=10,  # 10 Hz fundamental
        signal_amplitude=10,  # Signal strength
        signal_shape="rect",  # Rectangle = harmonics
        noise_amplitude=1,
    )  # Background noise

    # FFT analysis with windowing (1 second windows, 50% overlap)
    fft = gp.FFT(
        window_size=fs,  # 250 samples = 1 sec window
        overlap=0.5,  # 50% overlap for smooth updates
        window_function="hamming",
    )  # Reduce spectral leakage

    # Frequency domain visualization (spectrum analyzer)
    scope = gp.SpectrumScope(amplitude_limit=20)  # Y-axis: 0-20 dB

    # Connect processing chain: source -> FFT -> spectrum display
    p.connect(source, fft)
    p.connect(fft, scope)

    # Add spectrum scope to application window
    app.add_widget(scope)

    # Start pipeline and run application
    p.start()  # Begin signal processing
    app.run()  # Show GUI and start main loop
    p.stop()  # Clean shutdown
