"""
Unicorn Hybrid Black - EEG plus accelerometer, gyroscope and aux

With every optional stream on the device delivers 17 channels: 8 EEG,
3 accelerometer, 3 gyroscope, 3 auxiliary (battery, counter,
validation). A Router splits them so only the EEG is filtered; a second
Router recombines them for one scope, where the non-EEG channels rail
against the ±50 µV limit.

Requires: a paired Unicorn Hybrid Black and Unicorn Suite (Windows)
Run: python example_devices_hybrid_black.py
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
        include_aux=True,  # Battery/counter/validation, channels 15-17
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
