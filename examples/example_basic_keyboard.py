"""Keyboard events merged into a live signal and shown as markers.

Router concatenates the generator's 8 channels with the Keyboard's one
event channel, so a key code lands on channel 8 (0-based) -- which is
the channel the markers below are pinned to.

Requires: gpype[gui,devices]
Run: python example_basic_keyboard.py
"""
import gpype as gp

fs = 250  # Sampling frequency in Hz

if __name__ == "__main__":
    # Create the main application window
    app = gp.MainApp()

    # Create processing pipeline
    p = gp.Pipeline()

    # Generate synthetic 8-channel EEG-like signals
    source = gp.Generator(
        sampling_rate=fs,
        channel_count=8,  # 8 EEG channels
        signal_frequency=10,  # 10 Hz alpha rhythm
        signal_amplitude=10,  # Signal strength
        signal_shape="sine",  # Clean sine waves
        noise_amplitude=10,
    )  # Background noise

    # Capture keyboard input as event markers
    keyboard = gp.Keyboard()  # Arrow keys -> numerical event codes

    # Combine signal data (8 channels) + keyboard events (1 channel)
    router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])

    # Define colored markers for visual feedback (values are key codes)
    mk = gp.TimeSeriesScope.Markers
    markers = [
        mk(color="r", label="up", channel=8, value=38),
        mk(color="g", label="right", channel=8, value=39),
        mk(color="b", label="down", channel=8, value=40),
        mk(color="k", label="left", channel=8, value=37),
    ]

    # Real-time visualization with interactive markers
    scope = gp.TimeSeriesScope(
        amplitude_limit=30,  # Y-axis range
        time_window=10,  # 10 seconds history
        markers=markers,
    )  # Event visualization

    # Connect processing chain: signals + events -> combined display
    p.connect(source, router["in1"])  # Signal data -> Router input 1
    p.connect(keyboard, router["in2"])  # Keyboard events -> Router input 2
    p.connect(router, scope)  # Combined data -> Display

    # Add scope to application window
    app.add_widget(scope)

    # Start interactive signal processing
    p.start()  # Begin processing (press arrow keys for events)
    app.run()  # Show GUI and start main loop
    p.stop()  # Clean shutdown when window closes
