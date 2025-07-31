"""
Basic Bandstop (Notch) Filter Example - Power Line Interference Removal

This example demonstrates how to use a bandstop filter (also known as a notch
filter) to remove power line interference from a signal while preserving the
rest of the frequency spectrum. This is a common preprocessing step in BCI
and EEG signal processing.

What this example shows:
- Creating a synthetic signal with 50 Hz power line interference and noise
- Applying a bandstop filter (48-52 Hz) to remove the power line component
- Real-time visualization showing clean signal with interference removed
- Basic pipeline construction: Generator -> Bandstop -> Scope

Expected output:
When you run this example, you'll see that the strong 50 Hz power line
interference is completely removed from the signal, leaving primarily
the background noise. This demonstrates the practical application of bandstop
filters for cleaning BCI signals contaminated by electrical interference.

Usage:
    python example_basic_bandstop.py

Real-world application:
    Power line interference (50 Hz in Europe, 60 Hz in North America) is one
    of the most common artifacts in EEG/BCI recordings. This example shows
    how to effectively remove it while preserving the neural signals of
    interest.

Note:
    Compare this with example_basic_bandpass.py to see the complementary
    effect:
    - Bandpass: keeps specific frequencies, removes everything else
    - Bandstop: removes specific frequencies, keeps everything else
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
        signal_frequency=50,
        signal_amplitude=100,
        signal_shape="sine",
        noise_amplitude=10,
    )

    # bandstop
    filter = gp.Bandstop(f_lo=48, f_hi=52)

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
