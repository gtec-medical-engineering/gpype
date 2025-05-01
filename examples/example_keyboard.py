from scipy.signal import butter
import gpype as gp
fs = 250

if __name__ == '__main__':

    # main app
    app = gp.MainApp()

    # pipeline
    p = gp.Pipeline()

    # white noise source
    noise_source = gp.NoiseGenerator(sampling_rate=fs,
                                     channel_count=8,
                                     variance=30)

    # keystroke detector
    keyboard_source = gp.KeyboardCapture()

    # bandpass filter
    f_lo = 0.5
    f_hi = 30
    b, a = butter(N=2, Wn=[f_lo / (fs / 2), f_hi / (fs / 2)], btype='bandpass')
    filter = gp.LTIFilter(b=b, a=a)

    # scope
    markers = [{'channel': 8,
                'value': 38,
                'color': 'r',
                'label': 'up'},
               {'channel': 8,
                'value': 39,
                'color': 'b',
                'label': 'right'},
               {'channel': 8,
                'value': 40,
                'color': 'k',
                'label': 'down'},
               {'channel': 8,
                'value': 37,
                'color': 'm',
                'label': 'left'}
               ]
    scope = gp.TimeSeriesScope(amplitude_limit=30,
                               time_window=10,
                               markers=markers)

    # merge
    router = gp.Router(input_selector=[gp.Router.ALL, gp.Router.ALL])

    # file sink
    sink = gp.FileSink(file_name='test.csv')

    p.connect(noise_source, filter)
    p.connect_ports(filter, "out", router, "in1")
    p.connect_ports(keyboard_source, "out", router, "in2")
    p.connect(router, scope)
    p.connect(router, sink)

    # add widgets
    app.add_widget(scope)

    # start pipeline and main app
    p.start()
    app.run()
    p.stop()
