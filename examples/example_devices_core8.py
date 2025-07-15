"""
BCI Core-8 Device Example - Real-time EEG Acquisition and Processing

This example demonstrates how to connect to and process real-time EEG data from
a g.tec BCI Core-8 amplifier system. It showcases professional EEG signal
processing with standard filtering techniques commonly used in clinical and
research BCI applications.

This example demonstrates how to connect to and process real-time EEG data from
a g.tec BCI Core-8 amplifier system. It showcases EEG signal processing with
standard filtering techniques commonly used in clinical and research BCI
applications.

What this example shows:
- Real-time data acquisition from BCI Core-8 hardware
- Bandpass filtering for EEG frequency band selection
- Power line interference removal with dual notch filters
- Real-time visualization of clean EEG signals
- Hardware integration with g.Pype framework

Hardware requirements:
- g.tec BCI Core-8 EEG amplifier system

Expected behavior:
When you run this example:
- Connects to BCI Core-8 amplifier automatically
- Displays real-time EEG from 8 channels
- Shows clean, filtered signals suitable for analysis
- Amplitude range: ±50 µV (typical EEG range)
- Time window: 10 seconds of continuous data
- Real-time updates at amplifier sampling rate

Signal processing pipeline:
1. Raw EEG acquisition (8 channels, 250 Hz)
2. Bandpass filtering (1-30 Hz) - standard EEG band
3. 50 Hz notch filter - removes European power line noise
4. 60 Hz notch filter - removes American power line noise
5. Real-time visualization

Filtering rationale:
- Bandpass (1-30 Hz): Preserves main EEG rhythms while removing:
  * DC drift and very low frequencies (<1 Hz)
  * High-frequency noise and muscle artifacts (>30 Hz)
- Notch filters: Remove power line interference at:
  * 50 Hz (Europe, Asia, Africa, Australia)
  * 60 Hz (North America, parts of South America)

Real-world applications:
- Clinical EEG monitoring and diagnosis
- BCI system development and testing
- Neurofeedback training applications
- Cognitive state monitoring research
- Sleep study and analysis
- Seizure detection systems
- Attention and meditation training

Usage:
    1. Mount BCI Core-8 cap/electrodes
    2. Power on BCI Core-8
    4. Run: python example_devices_core8.py
    5. Monitor real-time EEG signals

Note:
    This example provides the foundation for all BCI applications
    requiring real-time EEG data acquisition and processing.
"""


import gpype as gp

# Sampling rate (hardware-dependent, typically 256 Hz for BCI Core-8)
fs = 250

if __name__ == '__main__':

    # Initialize main application for GUI and device management
    app = gp.MainApp()

    # Create real-time processing pipeline for EEG data
    p = gp.Pipeline()

    # === HARDWARE DATA SOURCE ===
    # BCI Core-8: Professional 8-channel EEG amplifier
    # Automatically detects and connects to available hardware
    # Provides high-quality, low-noise EEG signals at ~256 Hz
    source = gp.BCICore8()

    # === SIGNAL CONDITIONING STAGE ===
    # Bandpass filter: Extract standard EEG frequency range
    # 1-30 Hz preserves all major brain rhythms while removing:
    # - DC drift and movement artifacts (<1 Hz)
    # - EMG muscle artifacts and high-frequency noise (>30 Hz)
    bandpass = gp.Bandpass(f_lo=1,    # High-pass: remove DC and slow drift
                           f_hi=30)   # Low-pass: remove muscle artifacts

    # === POWER LINE INTERFERENCE REMOVAL ===
    # Notch filter for 50 Hz power line noise (European standard)
    # 48-52 Hz range accounts for slight frequency variations
    notch50 = gp.Bandstop(f_lo=48,   # Lower bound of 50 Hz notch
                          f_hi=52)   # Upper bound of 50 Hz notch

    # Notch filter for 60 Hz power line noise (American standard)
    # 58-62 Hz range accounts for slight frequency variations
    # Both filters ensure compatibility with different power systems
    notch60 = gp.Bandstop(f_lo=58,   # Lower bound of 60 Hz notch
                          f_hi=62)   # Upper bound of 60 Hz notch

    # === REAL-TIME VISUALIZATION ===
    # Professional EEG scope with clinical amplitude scaling
    # 50 µV range covers typical EEG signal amplitudes
    # 10-second window provides good temporal context
    scope = gp.TimeSeriesScope(amplitude_limit=50,  # ±50 µV range
                               time_window=10)      # 10-second display

    # === PIPELINE CONNECTIONS ===
    # Create signal processing chain: Hardware → Filtering → Visualization
    # Order matters: bandpass first, then notch filters, finally display

    # Connect hardware source to initial bandpass filter
    p.connect(source, bandpass)

    # Connect bandpass output to first notch filter (50 Hz)
    p.connect(bandpass, notch50)

    # Connect first notch to second notch filter (60 Hz)
    p.connect(notch50, notch60)

    # Connect final filtered signal to visualization scope
    p.connect(notch60, scope)

    # === APPLICATION SETUP ===
    # Add visualization widget to main application window
    app.add_widget(scope)

    # === EXECUTION ===
    # Start real-time data acquisition and processing
    p.start()    # Initialize hardware and begin data flow
    app.run()    # Start GUI event loop (blocks until window closes)
    p.stop()     # Clean shutdown: stop hardware and close connections
