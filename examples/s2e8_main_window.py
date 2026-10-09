"""Season 2, Episode 8: the main window's caption, grid, theme and sleep."""

import gpype as gp

if __name__ == "__main__":

    # A caption for the title bar, a 2 x 2 grid for the widgets, the
    # system's light or dark theme, and the screen allowed to sleep
    app = gp.MainApp(caption="Alpha Monitor", grid_size=[2, 2],
                     theme="system", prevent_sleep=False)
    p = gp.Pipeline()

    source = gp.Generator(signal_amplitude=10, noise_amplitude=5)
    alpha = gp.Bandpass(f_lo=8, f_hi=12)
    fft = gp.FFT(window_size=250)

    raw = gp.TimeSeriesScope(amplitude_limit=30)
    filtered = gp.TimeSeriesScope(amplitude_limit=30)
    spectrum = gp.SpectrumScope(amplitude_limit=20)

    p.connect(source, raw)
    p.connect(source, alpha)
    p.connect(alpha, filtered)
    p.connect(source, fft)
    p.connect(fft, spectrum)

    # Grid positions count from 1, row by row: the raw scope takes the
    # whole top row, the other two share the bottom one
    app.add_widget(raw, grid_positions=[1, 2])
    app.add_widget(filtered, grid_positions=[3])
    app.add_widget(spectrum, grid_positions=[4])

    p.start()
    app.run()
    p.stop()
    p.close()
