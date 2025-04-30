import gpype as gp
from scipy.signal import butter

fs = 250

if __name__ == '__main__':

    app = gp.MainApp()
    p = gp.Pipeline()

    amp = gp.BCICore8(channel_count=8)

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

    scope = gp.TimeSeriesScope(amplitude_limit=50,
                               time_window=10)
    sink = gp.FileSink(file_name='test.csv')

    app.add_widget(scope)
    p.connect(amp, f1)
    p.connect(f1, f2)
    p.connect(f2, scope)
    p.connect(f2, sink)
    p.start()
    app.run()
    p.stop()
