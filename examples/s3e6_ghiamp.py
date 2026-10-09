"""Season 3, Episode 6: a g.HIamp streamed into a filtered scope."""

import gpype as gp

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    # Opens the device here, through the GDS service. The rate must be
    # one the connected g.HIamp lists; anything else is refused
    source = gp.GHIamp(sampling_rate=1200, channel_count=8)

    # 1-30 Hz keeps the brain rhythms and drops the drift below and the
    # muscle activity above; two notches cover either mains standard
    bandpass = gp.Bandpass(f_lo=1, f_hi=30)
    notch50 = gp.Bandstop(f_lo=48, f_hi=52)
    notch60 = gp.Bandstop(f_lo=58, f_hi=62)

    scope = gp.TimeSeriesScope(amplitude_limit=50, time_window=10)

    p.connect(source, bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50, notch60)
    p.connect(notch60, scope)
    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()
