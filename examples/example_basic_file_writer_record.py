"""
Basic File Writer Example - Data Recording with Event Markers

This example demonstrates how to record data to CSV files while capturing
event markers from keyboard input. This is essential for BCI experiments
where you need to save both neural signals and behavioral events for
offline analysis.

What this example shows:
- Generating synthetic EEG-like signals (8 channels)
- Capturing keyboard events as experimental markers
- Combining signal data with event markers using Router
- Real-time visualization with color-coded event markers
- Saving all data (signals + events) to CSV file with timestamps

Expected output:
- Real-time scope showing 8-channel signals with event markers
- CSV file 'example_YYYYMMDD_HHMMSS.csv' containing:
  * Columns 0-7: Signal data from 8 channels
  * Column 8: Event markers (38=Up, 39=Right, 40=Down, 37=Left)
  * Automatic timestamp in filename prevents overwrites

Interactive controls:
- Arrow keys trigger colored markers in the display:
  * ↑ (Up): Red marker (value 38)
  * → (Right): Green marker (value 39)
  * ↓ (Down): Blue marker (value 40)
  * ← (Left): Black marker (value 37)

Real-world applications:
- BCI training data collection
- Event-related potential (ERP) experiments
- Motor imagery paradigm recording
- Behavioral experiment data logging
- Synchronizing neural and behavioral data

Technical details:
- Router combines 8 signal channels + 1 event channel = 9 total channels
- FileWriter automatically adds timestamps to prevent file overwrites
- Keyboard node converts key presses to numerical event codes
- Markers appear on channel 8 in both display and saved file

Usage:
    python example_basic_file_writer_record.py
    Press arrow keys to create event markers
    Close window to stop recording
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
    router = gp.Router(input_selector=[gp.Router.ALL, gp.Router.ALL])

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
    writer = gp.FileWriter(file_name="example_writer.csv")

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
