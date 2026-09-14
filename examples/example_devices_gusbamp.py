"""
g.USBamp - acquisition with the digital trigger kept separate

The trigger is split off by role, not by channel number, and reaches
its own scope unfiltered: a bandpass turns each edge into a decaying
oscillation that no longer reads as an event. That leaves the signal
path delayed by the filters' group delay and the trigger path not --
insert gp.Delay if you need them aligned sample-accurately.

Requires: a g.USBamp, with an event source wired to its trigger input
Run: python example_devices_gusbamp.py
"""
import gpype as gp

# Sampling rate (must be one the device supports)
fs = 256

# Number of channels to acquire, out of the 16 a g.USBamp provides
channel_count = 16

if __name__ == "__main__":

    # Initialize main application for GUI and device management
    app = gp.MainApp()

    # Create real-time processing pipeline
    p = gp.Pipeline()

    # === HARDWARE DATA SOURCE ===
    # g.USBamp: wired 16-channel biosignal amplifier, organised as four
    # groups of four channels. Tying all four groups to a common ground
    # and a common reference is the usual choice for an EEG montage; leave
    # them separate to record electrically isolated groups, e.g. EEG in
    # one group and EMG in another.
    #
    # enable_trigger adds the digital trigger as one extra channel after
    # the acquired ones. The parameter is enable_trigger here, not the
    # enable_di that g.HIamp uses.
    source = gp.GUSBamp(
        sampling_rate=fs,
        channel_count=channel_count,
        enable_trigger=True,
        common_ground=[True, True, True, True],
        common_reference=[True, True, True, True],
    )

    # === ROLE-BASED SPLIT ===
    # The amplifier declares its acquired channels as signal and the
    # appended digital input as a trigger, so neither path needs to know
    # which channel number the trigger ended up at.
    signal = gp.ChannelSelector(roles=gp.Constants.ChannelRoles.SIGNAL)
    trigger = gp.ChannelSelector(roles=gp.Constants.ChannelRoles.TRIGGER)

    # === SIGNAL CONDITIONING STAGE ===
    # Bandpass filter: keep the standard EEG range, removing DC drift and
    # movement below 1 Hz and muscle activity above 30 Hz.
    bandpass = gp.Bandpass(f_lo=1, f_hi=30)

    # Notch filter for 50 Hz power line noise (European standard)
    notch50 = gp.Bandstop(f_lo=48, f_hi=52)

    # === REAL-TIME VISUALIZATION ===
    # The biosignal scope, scaled for EEG
    scope = gp.TimeSeriesScope(amplitude_limit=50, time_window=10)

    # The trigger scope. The digital input's numeric level depends on the
    # amplifier's scaling and on how the input is wired, so raise this
    # limit if the trace rails, and lower it if the trace looks flat.
    trigger_scope = gp.TimeSeriesScope(amplitude_limit=2, time_window=10)

    # === PIPELINE CONNECTIONS ===
    # Biosignal path: select, filter, display
    p.connect(source, signal)
    p.connect(signal, bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50, scope)

    # Trigger path: taken straight from the amplifier, unfiltered
    p.connect(source, trigger)
    p.connect(trigger, trigger_scope)

    # === APPLICATION SETUP ===
    app.add_widget(scope)
    app.add_widget(trigger_scope)

    # === EXECUTION ===
    p.start()  # Initialize hardware and begin data flow
    app.run()  # Start GUI event loop (blocks until window closes)
    p.stop()  # Clean shutdown: stop hardware and release the device
    p.close()  # Release logging resources and the master timeline
