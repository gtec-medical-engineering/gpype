"""Season 5, Episode 3: one tone a hundred times, and its average response.

Windows only: it needs Paradigm Presenter, see Episode 1. The generator
stands in for an amplifier; to record a person, uncomment one amplifier
line and comment out the generator.
"""

import os

import gpype as gp

PARADIGM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "paradigms", "AEPSingleStim.xml")
CHANNELS = 8

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    # amp = gp.BCICore(channel_count=CHANNELS)
    # amp = gp.GNautilus(sampling_rate=250, channel_count=CHANNELS)
    amp = gp.Generator(sampling_rate=250, channel_count=CHANNELS,
                       signal_frequency=10, signal_amplitude=15,
                       noise_amplitude=10)

    # Filtered for display and epoching: 1 to 30 Hz, both mains notches
    bandpass = gp.Bandpass(f_lo=1, f_hi=30)
    notch50 = gp.Bandstop(f_lo=48, f_hi=52)
    notch60 = gp.Bandstop(f_lo=58, f_hi=62)
    p.connect(amp, bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50, notch60)

    # The paradigm's codes, and the M key (77) for a mark made by hand
    codes = gp.UDPReceiver()
    keyboard = gp.Keyboard()

    # Filtered signal, codes and key presses side by side on one scope
    mk = gp.TimeSeriesScope.Markers
    markers = [mk(color="r", label="stim", channel=CHANNELS, value=1),
               mk(color="c", label="M key", channel=CHANNELS + 1, value=77)]
    scope = gp.TimeSeriesScope(amplitude_limit=50, time_window=10,
                               markers=markers,
                               hidden_channels=[CHANNELS, CHANNELS + 1])
    shown = gp.Router(input_channels=[gp.Router.ALL] * 3)
    p.connect(notch60, shown["in1"])
    p.connect(codes, shown["in2"])
    p.connect(keyboard, shown["in3"])
    p.connect(shown, scope)

    # The recording keeps the unfiltered signal, with the same two beside
    # it, in a file named after the paradigm
    kept = gp.Router(input_channels=[gp.Router.ALL] * 3)
    p.connect(amp, kept["in1"])
    p.connect(codes, kept["in2"])
    p.connect(keyboard, kept["in3"])
    p.connect(kept, gp.CsvWriter(file_name="AEPSingleStim.csv"))

    # An epoch around every code 1, averaged as they arrive
    trigger = gp.Trigger(time_pre=0.2, time_post=0.7, target=1)
    average = gp.TriggerScope(amplitude_limit=5)
    p.connect(notch60, trigger)
    p.connect(codes, trigger[gp.Trigger.PORT_TRIGGER])
    p.connect(trigger, average)

    presenter = gp.ParadigmPresenter(PARADIGM)
    app.add_widget(presenter)
    app.add_widget(scope)
    app.add_widget(average)

    p.start()
    app.run()
    p.stop()
    p.close()
