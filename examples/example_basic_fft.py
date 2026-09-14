"""Show the live spectrum of a 10 Hz rectangular wave.

A rectangular wave carries odd harmonics, so peaks appear at 30, 50 and
70 Hz as well as at the fundamental -- a pure sine would show one peak.
The 250-sample window at 250 Hz gives 1 Hz resolution, and the 50%
overlap updates the display twice per window.

Requires: pip install gpype[gui]
Run: python example_basic_fft.py
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
