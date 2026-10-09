"""Season 2, Episode 6: the live spectrum of a 10 Hz rectangular wave."""

import gpype as gp

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    # A rectangular wave carries odd harmonics: 10, 30, 50, 70 Hz ...
    source = gp.Generator(sampling_rate=250, channel_count=2,
                          signal_frequency=10, signal_amplitude=10,
                          signal_shape="rect", noise_amplitude=1)

    # One spectrum per 250 samples (bins 1 Hz apart), a new one every 125
    fft = gp.FFT(window_size=250, overlap=0.5, window_function="hamming")
    scope = gp.SpectrumScope(amplitude_limit=20)

    p.connect(source, fft)
    p.connect(fft, scope)
    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()
