"""Season 2, Episode 3: an Equation multiplies two signals, one squares one."""

import gpype as gp

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    # Noise standing in for EEG, and a slow sine that will modulate it
    noise = gp.Generator(name="noise", sampling_rate=250, channel_count=8,
                         noise_amplitude=5)
    modulator = gp.Generator(name="modulator", sampling_rate=250,
                             channel_count=1, signal_frequency=0.5,
                             signal_amplitude=1)

    # n * (1 + m): the names in the expression are the input ports
    multiplier = gp.Equation("n * (1 + m)", name="multiplier")

    # Keep the alpha band, then square it: its power, sample by sample
    alpha = gp.Bandpass(f_lo=8, f_hi=12, name="alpha")
    power = gp.Equation("in**2", name="power")

    # One lane per stage, top to bottom
    merger = gp.Router(input_channels={"noise": [0], "modulator": [0],
                                       "multiplier": [0], "alpha": [0],
                                       "power": [0]},
                       output_channels=[gp.Router.ALL])
    scope = gp.TimeSeriesScope(amplitude_limit=10, time_window=10)

    p.connect(noise, multiplier["n"])
    p.connect(modulator, multiplier["m"])
    p.connect(multiplier, alpha)
    p.connect(alpha, power)

    p.connect(noise, merger["noise"])
    p.connect(modulator, merger["modulator"])
    p.connect(multiplier, merger["multiplier"])
    p.connect(alpha, merger["alpha"])
    p.connect(power, merger["power"])
    p.connect(merger, scope)
    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()
