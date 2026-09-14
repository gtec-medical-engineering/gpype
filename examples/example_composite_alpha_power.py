"""Alpha power from synthetic EEG, every stage shown side by side.

Noise is amplitude-modulated at 0.5 Hz, bandpassed to 8-12 Hz, squared,
smoothed, decimated to 5 Hz, and interpolated back up so the Router can
display it beside the full-rate stages. Router requires every input to
agree on sampling rate *and* frame size, which is why the feature returns
through an Interpolator instead of being shown at 5 Hz.

Requires: the gui extra (TimeSeriesScope)
Run: python example_composite_alpha_power.py
"""
import gpype as gp

# Sampling rate configuration for realistic EEG simulation
fs = 250  # 250 Hz - standard EEG sampling rate

if __name__ == "__main__":

    # Initialize main application for GUI event handling
    app = gp.MainApp()

    # Create processing pipeline for alpha power analysis
    p = gp.Pipeline()

    # === SIGNAL GENERATION STAGE ===
    # Generate 8-channel background noise simulating baseline EEG activity
    noise = gp.Generator(
        sampling_rate=fs,
        channel_count=8,
        noise_amplitude=5,  # 5 µV RMS noise level
        # Explicit, and it has to be: a source left without one picks a
        # frame size from its rate (4 at 250 Hz), while the decimated
        # branch comes back at 1 -- and Router refuses inputs whose
        # frame sizes disagree. Stating 1 everywhere makes them agree.
        frame_size=1,
    )

    # Generate low-frequency modulation signal (0.5 Hz sine wave)
    # This simulates natural alpha power fluctuations
    modulator = gp.Generator(
        sampling_rate=fs,
        channel_count=1,
        signal_frequency=0.5,  # 0.5 Hz modulation
        signal_amplitude=1,  # Modulation depth
        signal_shape="sine",
        frame_size=1,  # Matches the noise source; see above.
    )

    # === SIGNAL MODULATION STAGE ===
    # Apply amplitude modulation: noise × (1 + modulation)
    # Creates realistic EEG with time-varying alpha power
    multiplier = gp.Equation("n * (1 + m)")

    # === ALPHA BAND FILTERING STAGE ===
    # Extract alpha frequency band (8-12 Hz) from modulated signal
    # Standard alpha band definition for EEG analysis
    alpha_filter = gp.Bandpass(f_lo=8, f_hi=12)  # Upper alpha

    # === POWER ANALYSIS STAGE ===
    # Compute instantaneous power using signal squaring
    # Power = signal² provides envelope of alpha activity
    power = gp.Equation("in**2")

    # === TEMPORAL SMOOTHING STAGE ===
    # Apply moving average to stabilize power estimates
    # 125 samples = 0.5 seconds at 250 Hz sampling rate
    moving_average = gp.MovingAverage(window_size=125)

    # === DATA REDUCTION STAGE ===
    # Reduce data rate for efficient visualization and storage
    # Factor 50: 250 Hz → 5 Hz (adequate for alpha power tracking)
    decimator = gp.Decimator(decimation_factor=50)

    # Bring the 5 Hz feature back up to the signal rate so it can be
    # displayed alongside the full-rate stages below. The Router needs
    # every input to agree on sampling rate *and* frame size, and
    # Interpolator satisfies both by emitting 50 frames per input frame
    # rather than one frame 50 rows long. "hold" repeats the last
    # computed value, which is what a windowed estimate actually is --
    # see gp.Interpolator for why that is the default.
    upsampler = gp.Interpolator(interpolation_factor=50)

    # === VISUALIZATION ROUTING STAGE ===
    # Router combines all processing stages for comparative analysis
    # Allows simultaneous viewing of raw, filtered, and processed signals
    merger = gp.Router(
        input_channels={
            "noise": [0],  # Raw noise
            "modulator": [0],  # Modulation
            "multiplier": [0],  # Modulated
            "alpha_filter": [0],  # Alpha-filtered
            "power": [0],  # Power signal
            "moving_average": [0],  # Smoothed power
            "upsampled": [0],  # Decimated feature, back at 250 Hz
        },  # Final output
        output_channels=[gp.Router.ALL],
    )

    # === REAL-TIME VISUALIZATION ===
    # Multi-channel scope for real-time signal monitoring
    # 10-second time window with ±10 µV amplitude range
    scope = gp.TimeSeriesScope(
        amplitude_limit=10, time_window=10  # ±10 µV display range
    )  # 10-second time window

    # === PIPELINE CONNECTIONS ===
    # Connect main processing chain: noise → modulation → filter → power
    p.connect(noise, multiplier["n"])  # Noise input to multiplier
    p.connect(modulator, multiplier["m"])  # Modulation input to multiplier
    p.connect(multiplier, alpha_filter)  # Modulated signal to alpha filter
    p.connect(alpha_filter, power)  # Filtered signal to power analysis
    p.connect(power, moving_average)  # Power to temporal smoothing
    p.connect(moving_average, decimator)  # Smoothed power to decimation
    p.connect(decimator, upsampler)  # Decimated feature back to 250 Hz

    # Connect all processing stages to router for visualization
    p.connect(noise, merger["noise"])  # Stage 1: Raw noise
    p.connect(modulator, merger["modulator"])  # Stage 2: Modulation
    p.connect(multiplier, merger["multiplier"])  # Stage 3: Modulated
    p.connect(alpha_filter, merger["alpha_filter"])  # Stage 4: Filtered
    p.connect(power, merger["power"])  # Stage 5: Power
    p.connect(moving_average, merger["moving_average"])  # Stage 6: Smoothed
    p.connect(upsampler, merger["upsampled"])  # Stage 7: Final

    # Connect router output to visualization scope
    p.connect(merger, scope)

    # === APPLICATION SETUP ===
    # Add visualization widget to main application window
    app.add_widget(scope)

    # === EXECUTION ===
    # Start pipeline processing and run application event loop
    p.start()  # Begin real-time signal processing
    app.run()  # Start GUI and block until window closes
    p.stop()  # Clean shutdown of processing pipeline
