"""
Unicorn Hybrid Black Device Example - Real-time EEG Acquisition and Processing

This example demonstrates how to connect to and process real-time EEG data from
a g.tec Unicorn Hybrid Black amplifier system. It showcases EEG
signal processing with standard filtering techniques commonly used in clinical
and research BCI applications.

What this example shows:
- Real-time data acquisition from Unicorn Hybrid Black hardware
- Bandpass filtering for EEG frequency band selection
- Power line interference removal with notch filters
- Real-time visualization of EEG signals
- Hardware integration with g.Pype framework

Hardware requirements:
- g.tec Unicorn Hybrid Black EEG amplifier
- Windows operating system
- Unicorn Suite installed (including Python API)

Expected behavior:
When you run this example:
- Connects to Unicorn Hybrid Black amplifier automatically via Bluetooth
- Displays real-time EEG from 8 channels
- Shows filtered signals in real-time scope
- Amplitude range: ±50 µV (typical EEG range)
- Time window: 10 seconds of continuous data

Signal processing pipeline:
1. Raw EEG acquisition (8 channels, 250 Hz)
2. Bandpass filtering (1-30 Hz) - standard EEG band
3. 50 Hz notch filter - removes European power line noise
4. 60 Hz notch filter - removes American power line noise
5. Real-time visualization

Real-world applications:
- Mobile EEG monitoring and research
- BCI system development and testing
- Neurofeedback training applications
- Cognitive state monitoring research
- Consumer neurotechnology applications
- Attention and meditation training

Usage:
    1. Pair Unicorn Hybrid Black via Bluetooth
    2. Power on the device
    3. Run: python example_devices_hybrid_black.py
    4. Monitor real-time EEG signals

Note:
    This example provides the foundation for all BCI applications
    requiring real-time EEG data acquisition and processing with
    the Unicorn Hybrid Black wireless amplifier.
"""
import gpype as gp

# Sampling rate (hardware-dependent, fixed at 250 Hz for Unicorn Hybrid Black)
fs = 250

if __name__ == "__main__":

    # Initialize main application for GUI and device management
    app = gp.MainApp()

    # Create real-time processing pipeline for EEG data
    p = gp.Pipeline()

    # === HARDWARE DATA SOURCE ===
    # Unicorn Hybrid Black: Wireless 8-channel EEG amplifier
    # Automatically detects and connects to available hardware via Bluetooth
    # Provides high-quality, low-noise EEG signals at 250 Hz
    #
    # Optional sensor channels:
    #   include_accel: Add 3 accelerometer channels (X, Y, Z)
    #   include_gyro:  Add 3 gyroscope channels (X, Y, Z)
    #   include_aux:   Add auxiliary channels (battery, counter, validation)
    #
    source = gp.HybridBlack(
        include_accel=True,  # Enable accelerometer (channels 9-11)
        include_gyro=True,   # Enable gyroscope (channels 12-14)
        include_aux=True,    # Enable battery/counter/validation (channels 15-17)
    )

    splitter = gp.Router(input_channels=gp.Router.ALL,
                         output_channels={"EEG": range(8),
                                          "ACC": [8, 9, 10],
                                          "GYRO": [11, 12, 13],
                                          "AUX": [14, 15, 16]})

    # === SIGNAL CONDITIONING STAGE ===
    # Bandpass filter: Extract standard EEG frequency range
    # 1-30 Hz preserves all major brain rhythms while removing:
    # - DC drift and movement artifacts (<1 Hz)
    # - EMG muscle artifacts and high-frequency noise (>30 Hz)
    bandpass = gp.Bandpass(
        f_lo=1, f_hi=30  # High-pass: remove DC and slow drift
    )  # Low-pass: remove muscle artifacts

    # === POWER LINE INTERFERENCE REMOVAL ===
    # Notch filter for 50 Hz power line noise (European standard)
    # 48-52 Hz range accounts for slight frequency variations
    notch50 = gp.Bandstop(
        f_lo=48, f_hi=52  # Lower bound of 50 Hz notch
    )  # Upper bound of 50 Hz notch

    # Notch filter for 60 Hz power line noise (American standard)
    # 58-62 Hz range accounts for slight frequency variations
    # Both filters ensure compatibility with different power systems
    notch60 = gp.Bandstop(
        f_lo=58, f_hi=62  # Lower bound of 60 Hz notch
    )  # Upper bound of 60 Hz notch

    combiner = gp.Router(input_channels={"EEG": gp.Router.ALL,
                                         "ACC": gp.Router.ALL,
                                         "GYRO": gp.Router.ALL,
                                         "AUX": gp.Router.ALL},
                         output_channels=gp.Router.ALL)

    # === REAL-TIME VISUALIZATION ===
    # Professional EEG scope with clinical amplitude scaling
    # 50 µV range covers typical EEG signal amplitudes
    # 10-second window provides good temporal context
    scope = gp.TimeSeriesScope(
        amplitude_limit=50, time_window=10  # ±50 µV range
    )  # 10-second display

    # === PIPELINE CONNECTIONS ===
    # Create signal processing chain: Hardware → Filtering → Visualization
    # Order matters: bandpass first, then notch filters, finally display

    p.connect(source, splitter)  # Split raw data into EEG, ACC, GYRO, AUX

    # Connect hardware source to initial bandpass filter

    p.connect(splitter["EEG"], bandpass)

    # Connect bandpass output to first notch filter (50 Hz)
    p.connect(bandpass, notch50)

    # Connect first notch to second notch filter (60 Hz)
    p.connect(notch50, notch60)

    # Connect final filtered signal to visualization scope
    p.connect(notch60, combiner["EEG"])  # Send filtered EEG to combiner
    p.connect(splitter["ACC"], combiner["ACC"])  # Send ACC to combiner
    p.connect(splitter["GYRO"], combiner["GYRO"])  # Send GYRO to combiner
    p.connect(splitter["AUX"], combiner["AUX"])  # Send AUX to combiner
    p.connect(combiner, scope)  # Display combined signals in scope
    # === APPLICATION SETUP ===
    # Add visualization widget to main application window
    app.add_widget(scope)

    # === EXECUTION ===
    # Start real-time data acquisition and processing
    p.start()  # Initialize hardware and begin data flow
    app.run()  # Start GUI event loop (blocks until window closes)
    p.stop()  # Clean shutdown: stop hardware and close connections
