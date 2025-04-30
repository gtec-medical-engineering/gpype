import gpype as gp
from scipy.signal import butter

fs = 250

if __name__ == '__main__':

    # create app and pipeline
    app = gp.MainApp()
    p = gp.Pipeline()

    # g.Nautilus
    gds = gp.GNautilus(sampling_rate=250,
                       channel_count=8,
                       sensitivity=750000.0)

    # bandpass
    f_lo = 1
    f_hi = 30
    b, a = butter(N=2,
                  Wn=[f_lo / (fs / 2), f_hi / (fs / 2)],
                  btype='bandpass')
    f1 = gp.LTIFilter(b=b, a=a)

    # 50 Hz notch
    fc = 50.0  # center frequency
    bw = 2.0  # bandwidth
    b, a = butter(N=2,
                  Wn=[(fc - bw) / (fs / 2), (fc + bw) / (fs / 2)],
                  btype='bandstop')
    f2 = gp.LTIFilter(b=b, a=a)

    sqest = gp.SQEstimator()

    scope = gp.TimeSeriesScope(amplitude_limit=100,
                               time_window=10,
                               enable_sq=True)
    sink = gp.FileSink(file_name='test.csv')

    app.add_widget(scope)
    p.connect(gds, f1)
    p.connect(f1, f2)

    p.connect(f2, sqest)
    p.connect_ports(f2, 'out', scope, 'in')
    p.connect_ports(sqest, 'out', scope, 'sq')

    p.connect(f2, sink)

    p.start()
    app.run()
    p.stop()
