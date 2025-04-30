import gpype as gp
from scipy.signal import butter


fs = 250

if __name__ == '__main__':

    # main app
    app = gp.MainApp()

    # pipeline
    p = gp.Pipeline()

    # white noise source
    source = gp.NoiseGenerator(sampling_rate=fs,
                               channel_count=8,
                               variance=30)

    # bandpass filter
    f_lo = 1
    f_hi = 30
    b, a = butter(N=2, Wn=[f_lo / (fs / 2), f_hi / (fs / 2)], btype='bandpass')
    filter = gp.LTIFilter(b=b, a=a)

    # scope
    scope = gp.TimeSeriesScope(amplitude_limit=30,
                               time_window=10)

    # file sink
    sink = gp.FileSink(file_name='test.csv')

    p.connect(source, filter)
    p.connect(filter, scope)
    p.connect(filter, sink)

    # add widgets
    app.add_widget(scope)

    monitor = gp.PerformanceMonitor(pipeline=p)
    app.add_widget(monitor)

    # start pipeline and main app
    p.start()
    app.run()
    p.stop()
