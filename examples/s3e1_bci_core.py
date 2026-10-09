"""Season 3, Episode 1: a BCI Core streamed into a filtered scope."""

import gpype as gp

# A Core-8's electrode names are known to g.Pype. A Core-4 needs its own
# montage, or channel_count=4: the channel count is not discovered
MONTAGE = None
# MONTAGE = gp.Montage(["Fz", "Cz", "Pz", "Oz"])   # a Core-4

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    # The first amplifier found, at the rate it reports; serial= picks
    # one when several are in range
    source = gp.BCICore(montage=MONTAGE)

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
