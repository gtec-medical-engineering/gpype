"""
Composite Alpha Power Analysis Example - Real-time EEG Alpha Band Processing

This example demonstrates a complete alpha power analysis pipeline that
simulates ps EEG data with modulated alpha activity and processes it
through multiple signal processing stages. It showcases g.Pype features
including signal generation, filtering, power computation, and multi-channel
visualization.

What this example shows:
- Pseudo EEG signal simulation with modulated alpha activity
- Alpha band filtering (8-12 Hz) for brain rhythm analysis
- Power extraction using signal squaring
- Temporal smoothing with moving averages
- Data decimation for computational efficiency
- Multi-stage pipeline visualization with Router node
- Real-time processing of neurophysiological signals

Pipeline architecture:
1. Signal Generation: Creates 8-channel noise + alpha modulation
2. Alpha Filtering: Extracts 8-12 Hz frequency band
3. Power Analysis: Computes instantaneous power (signal squared)
4. Temporal Smoothing: Moving average for stable power estimates
5. Data Reduction: Decimation for efficient downstream processing
6. Multi-channel Display: Router combines all processing stages

Expected behavior:
When you run this example:
- Opens time-series visualization showing 7 processing stages
- Channel 1: Raw noisy signal with modulated alpha
- Channel 2: 0.5 Hz modulation signal (low-frequency envelope)
- Channel 3: Modulated signal (noise × (1 + modulation))
- Channel 4: Alpha-filtered signal (8-12 Hz band)
- Channel 5: Instantaneous alpha power (squared signal)
- Channel 6: Smoothed alpha power (moving average)
- Channel 7: Decimated power (reduced sampling rate)

Real-world applications:
- EEG alpha rhythm analysis for neurofeedback systems
- Brain state monitoring and cognitive load assessment
- Sleep stage detection using alpha power dynamics
- Attention and relaxation state classification
- Real-time BCI applications using alpha modulation
- Clinical EEG analysis for neurological assessment

Scientific background:
Alpha waves (8-12 Hz) are prominent EEG rhythms associated with:
- Relaxed wakefulness and eyes-closed states
- Attention regulation and cognitive processing
- Sensorimotor idle states and cortical inhibition
- Individual differences in cognitive performance
- Pathological changes in neurological disorders

Technical features:
- Realistic signal modeling with controlled alpha modulation
- Multi-stage filtering and power analysis pipeline
- Efficient data processing with decimation
- Real-time parameter monitoring across processing stages

Pipeline components explained:
- Generator: Creates controllable synthetic EEG-like signals
- Equation: Performs mathematical operations (modulation, power)
- Bandpass: Implements digital filtering for frequency selection
- MovingAverage: Temporal smoothing for stable estimates
- Decimator: Reduces data rate while preserving information
- Router: Combines multiple signals for comparative visualization
- TimeSeriesScope: Real-time multi-channel display

Usage:
    python example_composite_alpha_power.py

Prerequisites:
    - g.Pype framework with signal processing modules
    - Real-time visualization capabilities
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
        sampling_rate=fs, channel_count=8, noise_amplitude=5
    )  # 5 µV RMS noise level

    # Generate low-frequency modulation signal (0.5 Hz sine wave)
    # This simulates natural alpha power fluctuations
    modulator = gp.Generator(
        sampling_rate=fs,
        channel_count=1,
        signal_frequency=0.5,  # 0.5 Hz modulation
        signal_amplitude=1,  # Modulation depth
        signal_shape="sine",
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

    # Hold last value for stable display between updates
    hold = gp.Hold()

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
            "hold": [0],
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
    p.connect(decimator, hold)  # Decimated signal to hold buffer

    # Connect all processing stages to router for visualization
    p.connect(noise, merger["noise"])  # Stage 1: Raw noise
    p.connect(modulator, merger["modulator"])  # Stage 2: Modulation
    p.connect(multiplier, merger["multiplier"])  # Stage 3: Modulated
    p.connect(alpha_filter, merger["alpha_filter"])  # Stage 4: Filtered
    p.connect(power, merger["power"])  # Stage 5: Power
    p.connect(moving_average, merger["moving_average"])  # Stage 6: Smoothed
    p.connect(hold, merger["hold"])  # Stage 7: Final

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
