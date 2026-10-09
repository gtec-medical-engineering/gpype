"""Season 2, Episode 5: a Decimator and an Interpolator change the rate."""

import gpype as gp

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    # The chain of Episode 4. frame_size=1 on both sources: the decimated
    # branch comes back in frames of 1, and the Router needs every input
    # to agree on the frame size as well as the rate
    noise = gp.Generator(name="noise", sampling_rate=250, channel_count=8,
                         noise_amplitude=5, frame_size=1)
    modulator = gp.Generator(name="modulator", sampling_rate=250,
                             channel_count=1, signal_frequency=0.5,
                             signal_amplitude=1, frame_size=1)
    multiplier = gp.Equation("n * (1 + m)", name="multiplier")
    alpha = gp.Bandpass(f_lo=8, f_hi=12, name="alpha")
    power = gp.Equation("in**2", name="power")
    smoothed = gp.MovingAverage(window_size=125, name="smoothed")

    # One sample in 50: 250 Hz becomes 5 Hz, anti-alias filtered first
    decimator = gp.Decimator(decimation_factor=50, name="decimator")

    # Back to 250 Hz, each value repeated 50 times, so the Router can
    # show the 5 Hz feature beside the full-rate stages
    upsampler = gp.Interpolator(interpolation_factor=50, name="upsampler")

    merger = gp.Router(input_channels={"noise": [0], "modulator": [0],
                                       "multiplier": [0], "alpha": [0],
                                       "power": [0], "smoothed": [0],
                                       "decimated": [0]},
                       output_channels=[gp.Router.ALL])
    scope = gp.TimeSeriesScope(amplitude_limit=10, time_window=10)

    p.connect(noise, multiplier["n"])
    p.connect(modulator, multiplier["m"])
    p.connect(multiplier, alpha)
    p.connect(alpha, power)
    p.connect(power, smoothed)
    p.connect(smoothed, decimator)
    p.connect(decimator, upsampler)

    p.connect(noise, merger["noise"])
    p.connect(modulator, merger["modulator"])
    p.connect(multiplier, merger["multiplier"])
    p.connect(alpha, merger["alpha"])
    p.connect(power, merger["power"])
    p.connect(smoothed, merger["smoothed"])
    p.connect(upsampler, merger["decimated"])
    p.connect(merger, scope)
    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()
