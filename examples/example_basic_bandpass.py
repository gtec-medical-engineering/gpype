"""
Basic Bandpass Filter Example

This example demonstrates the fundamental concepts of g.Pype by creating a
simple signal processing pipeline that generates a synthetic signal, applies
bandpass filtering, and visualizes the results in real-time.

What this example shows:
- Creating a synthetic 10 Hz sine wave with background noise
- Applying a narrow bandpass filter (9-11 Hz) to isolate the target frequency
- Real-time visualization of the filtered signal using TimeSeriesScope
- Basic pipeline construction: Generator -> Bandpass -> Scope

Expected output:
When you run this example, you'll see a clean 10 Hz sine wave displayed in the
scope window, with most of the background noise filtered out. This demonstrates
how bandpass filters can isolate specific frequency components from noisy
signals, which is essential for BCI applications like detecting brain rhythms
or motor imagery patterns.

Usage:
    python example_basic_bandpass.py

Dependencies:
    - g.Pype framework
    - PyQt5/PySide for GUI visualization
"""

import gpype as gp

if __name__ == "__main__":

    # main app
    app = gp.MainApp()

    # pipeline
    p = gp.Pipeline()

    # signal generator
    source = gp.Generator(
        sampling_rate=250,
        channel_count=8,
        signal_frequency=10,
        signal_amplitude=10,
        signal_shape="sine",
        noise_amplitude=1,
    )

    # bandpass
    filter = gp.Bandpass(f_lo=9, f_hi=11)

    # scope
    scope = gp.TimeSeriesScope(amplitude_limit=30, time_window=10)

    # connect nodes
    p.connect(source, filter)
    p.connect(filter, scope)

    # add widgets
    app.add_widget(scope)

    # start pipeline and main app
    p.start()
    app.run()
    p.stop()
