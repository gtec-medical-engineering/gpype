"""Season 3, Episode 4: a Unicorn's EEG, motion and status in one pipeline."""

import gpype as gp

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    # 8 EEG channels at 250 Hz, plus the optional streams: 3 accelerometer,
    # 3 gyroscope, and battery, counter, validation. 17 channels in all
    source = gp.HybridBlack(include_accel=True, include_gyro=True,
                            include_aux=True)

    # Split them by stream. The counter (channel 15) is left out: a ramp
    # no fixed axis shows, and a lost sample does not show in it
    splitter = gp.Router(input_channels=gp.Router.ALL,
                         output_channels={"EEG": list(range(8)),
                                          "ACC": [8, 9, 10],
                                          "GYRO": [11, 12, 13],
                                          "BATTERY": [14],
                                          "VALIDATION": [16]})

    # Only the EEG is filtered
    bandpass = gp.Bandpass(f_lo=1, f_hi=30)
    notch50 = gp.Bandstop(f_lo=48, f_hi=52)
    notch60 = gp.Bandstop(f_lo=58, f_hi=62)

    # One scope per stream, each scaled for its own unit: uV, g, deg/s,
    # percent, and 1 or 0. Named: five scopes sharing the default name
    # could not be saved
    eeg_scope = gp.TimeSeriesScope(name="eeg", amplitude_limit=50,
                                   time_window=10)
    acc_scope = gp.TimeSeriesScope(name="acc", amplitude_limit=2,
                                   time_window=10)
    gyro_scope = gp.TimeSeriesScope(name="gyro", amplitude_limit=250,
                                    time_window=10)
    battery_scope = gp.TimeSeriesScope(name="battery", amplitude_limit=100,
                                       time_window=10)
    valid_scope = gp.TimeSeriesScope(name="validation", amplitude_limit=1,
                                     time_window=10)

    p.connect(source, splitter)
    p.connect(splitter["EEG"], bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50, notch60)
    p.connect(notch60, eeg_scope)
    p.connect(splitter["ACC"], acc_scope)
    p.connect(splitter["GYRO"], gyro_scope)
    p.connect(splitter["BATTERY"], battery_scope)
    p.connect(splitter["VALIDATION"], valid_scope)

    for scope in (eeg_scope, acc_scope, gyro_scope, battery_scope,
                  valid_scope):
        app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()
