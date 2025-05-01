import gpype as gp
from scipy.signal import butter

paradigm_root_folder = "examples/paradigms"

fs = 250

if __name__ == '__main__':

    # main app
    app = gp.MainApp(caption="test")

    # pipeline
    p = gp.Pipeline()

    # amplifier
    amp = gp.NoiseGenerator(variance=30,
                            channel_count=8,
                            sampling_rate=fs)

    # trigger receiver
    udp_receiver = gp.UDPReceiver()

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

    # scope
    markers = [{'channel': 8,
                'value': 1,
                'color': 'r',
                'label': '1'},
               {'channel': 8,
                'value': 2,
                'color': 'g',
                'label': '2'},
               {'channel': 8,
                'value': 3,
                'color': 'b',
                'label': '3'},
               {'channel': 8,
                'value': 99,
                'color': 'k',
                'label': 'Rest'},
               ]
    scope = gp.TimeSeriesScope(amplitude_limit=100,
                               time_window=10,
                               markers=markers,
                               hidden_channels=[8])

    # merge
    router = gp.Router(input_selector=[gp.Router.ALL, gp.Router.ALL])

    # file sink
    sink = gp.FileSink(file_name='test.csv')

    p.connect(amp, f1)
    p.connect(f1, f2)
    p.connect_ports(f2, "out", router, "in1")
    p.connect_ports(udp_receiver, "out", router, "in2")
    p.connect(router, scope)
    p.connect(router, sink)

    # add widgets
    presenter = gp.ParadigmPresenter(paradigm_root_folder)
    app.add_widget(presenter)
    app.add_widget(scope)

    # start pipeline and main app
    p.start()
    app.run()
    p.stop()
