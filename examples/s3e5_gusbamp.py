"""Season 3, Episode 5: a g.USBamp, its trigger on a scope of its own."""

import gpype as gp

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    # 16 channels in four groups of four. A common ground and reference
    # across the groups is the usual EEG montage; separate groups record
    # electrically isolated signals, EEG in one and EMG in another.
    # enable_trigger appends the digital input as one more channel
    source = gp.GUSBamp(sampling_rate=256, channel_count=16,
                        enable_trigger=True,
                        common_ground=[True, True, True, True],
                        common_reference=[True, True, True, True])

    # Split by role, not by channel number: the amplifier declares its
    # inputs as signal and the digital input as a trigger
    signal = gp.ChannelSelector(roles=gp.Constants.ChannelRoles.SIGNAL)
    trigger = gp.ChannelSelector(roles=gp.Constants.ChannelRoles.TRIGGER)

    # The filters would skip the trigger by role anyway; the split gives
    # each path a scope at its own scale
    bandpass = gp.Bandpass(f_lo=1, f_hi=30)
    notch50 = gp.Bandstop(f_lo=48, f_hi=52)

    scope = gp.TimeSeriesScope(name="eeg", amplitude_limit=50,
                               time_window=10)
    # The digital input's level depends on the amplifier's scaling and
    # on the wiring: raise this limit if the trace rails
    trigger_scope = gp.TimeSeriesScope(name="trigger", amplitude_limit=2,
                                       time_window=10)

    p.connect(source, signal)
    p.connect(signal, bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50, scope)
    p.connect(source, trigger)
    p.connect(trigger, trigger_scope)
    app.add_widget(scope)
    app.add_widget(trigger_scope)

    p.start()
    app.run()
    p.stop()
    p.close()
