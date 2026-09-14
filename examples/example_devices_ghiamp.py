"""
g.HIamp - real-time EEG acquisition, filtering and display

Bandpass 1-30 Hz, then notches at 50 and 60 Hz so the example runs on
either mains standard. fs must be a rate the connected g.HIamp lists as
supported; anything else is refused when the device opens.

Requires: a g.HIamp, and pip install gpype[all]
Run: python example_devices_ghiamp.py
"""
import gpype as gp

# Sampling rate (hardware-dependent, typically 256 Hz for g.HIamp)
fs = 1200

if __name__ == "__main__":

    # Initialize main application for GUI and device management
    app = gp.MainApp()

    # Create real-time processing pipeline for EEG data
    p = gp.Pipeline()

    # === HARDWARE DATA SOURCE ===
    # g.HIamp: Professional wired multi-channel EEG amplifier
    # Automatically detects and connects to available device
    # Provides high-quality, low-noise wired EEG signals at 1200 Hz
    source = gp.GHIamp(sampling_rate=fs, channel_count=8)  # 8 EEG channels

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
    p.start()  # Initialize hardware and begin data flow
    app.run()  # Start GUI event loop (blocks until window closes)
    p.stop()  # Clean shutdown: stop hardware and close connections
