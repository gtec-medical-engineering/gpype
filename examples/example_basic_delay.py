"""
Basic Delay Example - Signal Time Shifting and Synchronization

This example demonstrates how to use the Delay node to introduce controlled
time delays in signal processing pipelines. This is useful for synchronization,
timing alignment, and creating phase-shifted signals for analysis.

What this example shows:
- Generating a dual-channel 1 Hz sine wave signal
- Splitting the signal into two separate paths using Router
- Applying a 0.5-second delay (125 samples at 250 Hz) to one path
- Merging the original and delayed signals for comparison
- Visualizing both signals simultaneously to see the phase shift

Expected output:
When you run this example, you'll see two sine waves in the scope:
- Channel 1: Original signal
- Channel 2: Same signal delayed by 0.5 seconds (180° phase shift at 1 Hz)

The delayed signal will appear to "lag behind" the original, creating a clear
visual demonstration of the time delay effect.

Real-world applications:
- Signal synchronization between different data sources
- Creating reference signals for cross-correlation analysis
- Compensating for processing delays in real-time systems
- Phase shift analysis in BCI applications
- Latency compensation in multi-channel recordings

Usage:
    python example_basic_delay.py

Technical details:
- Sampling rate: 250 Hz
- Signal frequency: 1 Hz (easy to visualize delay effects)
- Delay: 125 samples = 0.5 seconds = 180° phase shift
- Pipeline: Generator -> Splitter -> [Path1: Direct, Path2: Delay]
            -> Merger -> Scope
"""


import gpype as gp


if __name__ == '__main__':

    # main app
    app = gp.MainApp()

    # pipeline
    p = gp.Pipeline()

    # signal generator
    source = gp.Generator(sampling_rate=250,
                          channel_count=2,
                          signal_frequency=1,
                          signal_amplitude=10,
                          signal_shape='sine',
                          noise_amplitude=1)

    # split signals
    splitter = gp.Router(input_selector=[gp.Router.ALL],
                         output_selector=[[0], [1]])

    # delay one signal by 125 samples (0.5 seconds at 250 Hz)
    delay = gp.Delay(num_samples=125)

    # merge signals back together
    merger = gp.Router(input_selector=[[0], [0]],
                       output_selector=[gp.Router.ALL])

    # scope
    scope = gp.TimeSeriesScope(amplitude_limit=30,
                               time_window=10)

    # connect nodes
    p.connect(source, splitter)
    p.connect(splitter["out1"], merger["in1"])
    p.connect(splitter["out2"], delay)
    p.connect(delay, merger["in2"])
    p.connect(merger, scope)

    # add widgets
    app.add_widget(scope)

    # start pipeline and main app
    p.start()
    app.run()
    p.stop()
