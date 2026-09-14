"""Record 8 generated channels plus keyboard markers to a CSV file.

The Router merges the 8 signal channels with the Keyboard's single event
channel, so marker codes land on channel 8 in both the scope and the
file. CsvWriter inserts a timestamp into the file name, so every run
writes a new file into the current directory. Arrow keys produce the
codes 37, 38, 39, 40 (left, up, right, down); close the window to stop.

Requires: pip install gpype[gui,devices]
Run: python example_basic_csv_writer_record.py
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
        signal_frequency=10,  # 10 Hz alpha-like rhythm
        signal_amplitude=10,  # Signal strength
        signal_shape="sine",  # Clean sine waves
        noise_amplitude=10,
    )  # Realistic noise level

    # Capture keyboard input as event markers
    keyboard = gp.Keyboard()  # Arrow keys -> event codes

    # Combine signal data (8 channels) + event data (1 channel) = 9 channels
    router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])

    # Define colored markers for visualization (values korrespond to arrows)
    mk = gp.TimeSeriesScope.Markers
    markers = [
        mk(color="r", label="up", channel=8, value=38),
        mk(color="g", label="right", channel=8, value=39),
        mk(color="b", label="down", channel=8, value=40),
        mk(color="k", label="left", channel=8, value=37),
    ]

    # Real-time display with event markers
    scope = gp.TimeSeriesScope(
        amplitude_limit=30,  # Y-axis range
        time_window=10,  # 10 seconds history
        markers=markers,
    )  # Show event markers

    # CSV file writer (auto-timestamped filename)
    writer = gp.CsvWriter(file_name="example_writer.csv")

    # Connect processing chain
    p.connect(source, router["in1"])  # Signal data -> Router input 1
    p.connect(keyboard, router["in2"])  # Event data -> Router input 2
    p.connect(router, scope)  # Combined data -> Display
    p.connect(router, writer)  # Combined data -> File

    # Add scope to application window
    app.add_widget(scope)

    # Start recording and visualization
    p.start()
    app.run()
    p.stop()
