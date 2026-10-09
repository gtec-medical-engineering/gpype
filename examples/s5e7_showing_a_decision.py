"""Season 5, Episode 7: a decision shown as a number, not as a trace."""

import gpype as gp

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    # 10 Hz alpha switched on for 2.5 s and off for 2.5 s, in noise
    alpha = gp.Generator(name="alpha", sampling_rate=250, channel_count=1,
                         signal_frequency=10, signal_amplitude=10)
    gate = gp.Generator(name="gate", sampling_rate=250, channel_count=1,
                        signal_frequency=0.2, signal_shape="rect",
                        signal_amplitude=1)
    noise = gp.Generator(name="noise", sampling_rate=250, channel_count=1,
                         noise_amplitude=3)
    eeg = gp.Equation("a * (1 + g) / 2 + n")
    p.connect(alpha, eeg["a"])
    p.connect(gate, eeg["g"])
    p.connect(noise, eeg["n"])

    # One spectrum per half second (window 1 s, overlap 0.5), reduced to
    # the alpha band's power
    fft = gp.FFT(window_size=250)
    power = gp.BandPower(bands=["alpha"])
    p.connect(eeg, fft)
    p.connect(fft, power)

    # The decision: 1 while alpha is present
    decision = gp.Threshold(level=20)
    p.connect(power, decision)

    # Measurement and decision side by side, named for the readout
    both = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])
    names = gp.ChannelLabeler(labels=["alpha power", "alpha present"])
    p.connect(power, both["in1"])
    p.connect(decision, both["in2"])
    p.connect(both, names)

    # The last four values beside the current one, one decimal
    readout = gp.ResultScope(name="decision", history=4, decimals=1)
    p.connect(names, readout)

    # The signal the decision is made from, for comparison
    scope = gp.TimeSeriesScope(name="eeg", amplitude_limit=30,
                               time_window=10)
    p.connect(eeg, scope)

    app.add_widget(scope)
    app.add_widget(readout)

    p.start()
    app.run()
    p.stop()
    p.close()
