from pathlib import Path
import os

import gpype as gp

parent_dir = os.path.dirname(os.path.abspath(__file__))
paradigm = os.path.join(parent_dir, "paradigms", "AEPSingleStim.xml")
sampling_rate = 250
channel_count = 8

if __name__ == "__main__":

    # Create main application & pipeline
    app = gp.MainApp()
    p = gp.Pipeline()

    # Amplifier: BCI Core-8 (uncomment if you want to use it)
    # amp = gp.BCICore8()

    # Amplifier: g.Nautilus (uncomment if you want to use it)
    # amp = gp.GNautilus(sampling_rate=sampling_rate,
    #                    channel_count=channel_count)

    # Signal generator (uncomment if you want to use it)
    amp = gp.Generator(
        sampling_rate=sampling_rate,
        channel_count=channel_count,
        signal_frequency=10,
        signal_amplitude=15,
        signal_shape="sine",
        noise_amplitude=10,
    )

    # Bandpass from 1 to 30 Hz
    bandpass = gp.Bandpass(f_lo=1, f_hi=30)

    # Bandstop (notch) filter for 50 and 60 Hz
    notch50 = gp.Bandstop(f_lo=48, f_hi=52)
    notch60 = gp.Bandstop(f_lo=58, f_hi=62)

    # Presenter trigger receiver
    trig_receiver = gp.UDPReceiver()

    # Trigger data
    trig_node = gp.Trigger(time_pre=0.2, time_post=0.7, target=1)

    # Keyboard capture
    key_capture = gp.Keyboard()

    # Time series scope
    mk = gp.TimeSeriesScope.Markers
    markers = [
        mk(color="#ff0000", label="Stim", channel=channel_count, value=1),
        mk(
            color="#0000ff", label="M Key", channel=channel_count + 1, value=77
        ),
    ]
    scope = gp.TimeSeriesScope(
        amplitude_limit=50, time_window=10, markers=markers
    )

    # Trigger scope
    trig_scope = gp.TriggerScope(amplitude_limit=5)

    # Merge signals for scope and data saving
    router_scope = gp.Router(
        input_channels=[gp.Router.ALL, gp.Router.ALL, gp.Router.ALL]
    )
    router_raw = gp.Router(
        input_channels=[gp.Router.ALL, gp.Router.ALL, gp.Router.ALL]
    )

    # File writer
    filename = Path(paradigm).stem
    writer = gp.CsvWriter(file_name=f"{filename}.csv")

    # connect amplifier to filter nodes
    p.connect(amp, bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50, notch60)

    # merge scope data
    p.connect(notch60, router_scope["in1"])
    p.connect(trig_receiver, router_scope["in2"])
    p.connect(key_capture, router_scope["in3"])

    # merge raw data
    p.connect(amp, router_raw["in1"])
    p.connect(trig_receiver, router_raw["in2"])
    p.connect(key_capture, router_raw["in3"])

    # connect inputs of time series scope and file writer
    p.connect(router_scope, scope)
    p.connect(router_raw, writer)

    # connect trigger node and scope
    p.connect(notch60, trig_node[gp.Constants.Defaults.PORT_IN])
    p.connect(trig_receiver, trig_node[gp.Trigger.PORT_TRIGGER])
    p.connect(trig_node, trig_scope)

    # Create main app and add widgets
    presenter = gp.ParadigmPresenter(paradigm)
    app.add_widget(presenter)
    app.add_widget(scope)
    app.add_widget(trig_scope)

    # start pipeline and main app
    p.start()
    app.run()
    p.stop()
