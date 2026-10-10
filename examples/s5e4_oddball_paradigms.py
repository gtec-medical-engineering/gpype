"""Season 5, Episode 4: a rare tone among frequent ones, two averages.

Windows only: it needs Paradigm Presenter, see Episode 1. The generator
stands in for an amplifier; to record a person, uncomment one amplifier
line and comment out the generator.
"""

import os

import gpype as gp

PARADIGM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "paradigms", "AEPOddball.xml")
CHANNELS = 8
TARGET = 1  # the rare tone's code
NONTARGET = 2  # the frequent one's

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

    # The presenter receives the codes; the M key (77) marks by hand
    presenter = gp.ParadigmPresenter(PARADIGM)
    keyboard = gp.Keyboard()

    # Filtered signal, codes and key presses side by side on one scope
    mk = gp.TimeSeriesScope.Markers
    markers = [mk(color="r", label="target", channel=CHANNELS,
                  value=TARGET),
               mk(color="g", label="nontarget", channel=CHANNELS,
                  value=NONTARGET),
               mk(color="c", label="M key", channel=CHANNELS + 1, value=77)]
    scope = gp.TimeSeriesScope(amplitude_limit=50, time_window=10,
                               markers=markers,
                               hidden_channels=[CHANNELS, CHANNELS + 1])
    shown = gp.Router(input_channels=[gp.Router.ALL] * 3)
    p.connect(notch60, shown["in1"])
    p.connect(presenter, shown["in2"])
    p.connect(keyboard, shown["in3"])
    p.connect(shown, scope)

    # The recording keeps the unfiltered signal, with the same two beside
    # it, in a file named after the paradigm
    kept = gp.Router(input_channels=[gp.Router.ALL] * 3)
    p.connect(amp, kept["in1"])
    p.connect(presenter, kept["in2"])
    p.connect(keyboard, kept["in3"])
    p.connect(kept, gp.CsvWriter(file_name="AEPOddball.csv"))

    # One trigger per code, and a scope with a curve per average and a
    # third for their difference
    rare = gp.Trigger(time_pre=0.2, time_post=0.7, target=TARGET)
    frequent = gp.Trigger(time_pre=0.2, time_post=0.7, target=NONTARGET)
    average = gp.TriggerScope(amplitude_limit=5,
                              plots=["target", "nontarget",
                                     "target-nontarget"])
    p.connect(notch60, rare)
    p.connect(presenter, rare[gp.Trigger.PORT_TRIGGER])
    p.connect(notch60, frequent)
    p.connect(presenter, frequent[gp.Trigger.PORT_TRIGGER])
    p.connect(rare, average["target"])
    p.connect(frequent, average["nontarget"])

    app.add_widget(presenter)
    app.add_widget(scope)
    app.add_widget(average)

    p.start()
    app.run()
    p.stop()
    p.close()
