"""Combining channels from two sources into one stream.

Router takes channels 0 and 1 from the sine generator and 3, 4, 5 from
the rectangular one, giving a 5-channel output. Input channels are
counted from 0; the scope labels its channels from 1.

Requires: gpype[gui]
Run: python example_basic_router.py
"""
import gpype as gp

fs = 250  # Sampling frequency in Hz

if __name__ == "__main__":
    # Create the main application window
    app = gp.MainApp()

    # Create processing pipeline
    p = gp.Pipeline()

    # Generate first signal source: 10 Hz sine waves
    source_sine = gp.Generator(
        sampling_rate=fs,
        channel_count=8,  # 8 channels
        signal_frequency=10,  # 10 Hz frequency
        signal_amplitude=10,  # High amplitude
        signal_shape="sine",
    )  # Smooth sine waves

    # Generate second signal source: 2 Hz rectangular waves
    source_square = gp.Generator(
        sampling_rate=fs,
        channel_count=8,  # 8 channels
        signal_frequency=2,  # 2 Hz frequency
        signal_amplitude=5,  # Lower amplitude
        signal_shape="rect",
    )  # Rectangular signal

    # Router for selective channel combination
    # Input 1: Take channels 0,1 from sine generator
    # Input 2: Take channels 3,4,5 from rectangular generator
    # Output: 5 channels total (2 sine + 3 rectangular)
    router = gp.Router(input_channels=[[0, 1], [3, 4, 5]])

    # Real-time visualization scope
    scope = gp.TimeSeriesScope(
        amplitude_limit=30, time_window=10  # Y-axis range
    )  # 10 seconds display

    # Connect the multi-source pipeline
    p.connect(source_sine, router["in1"])  # Sine waves -> Router input 1
    p.connect(source_square, router["in2"])  # Rectangular -> Router input 2
    p.connect(router, scope)  # Combined signals -> Display

    # Add scope to application window
    app.add_widget(scope)

    # Start multi-source signal processing
    p.start()  # Begin processing (observe mixed signal types)
    app.run()  # Show GUI and start main loop
    p.stop()  # Clean shutdown when window closes
