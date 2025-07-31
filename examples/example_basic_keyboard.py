"""
Basic Keyboard Example - Interactive Event Markers and User Input

This example demonstrates how to capture keyboard input and combine it with
signal data to create interactive BCI experiments. It shows how to use the
Keyboard node for real-time event marking and user interaction during data
visualization.

What this example shows:
- Generating synthetic 8-channel EEG-like signals
- Capturing keyboard input (arrow keys) as event markers
- Combining signal data with keyboard events using Router
- Real-time visualization with color-coded event markers
- Interactive user input during signal processing

Expected output:
When you run this example, you'll see:
- 8 channels of synthetic EEG signals in real-time
- Event markers appearing when you press arrow keys:
  * ↑ (Up): Red marker on channel 8
  * → (Right): Green marker on channel 8
  * ↓ (Down): Blue marker on channel 8
  * ← (Left): Black marker on channel 8
- Combined display of continuous signals and discrete events

Interactive controls:
Press arrow keys while the application is running to create event markers.
Each key press generates a numerical code that appears as a colored spike
in the display and gets combined with the signal data.

Real-world applications:
- BCI paradigm development and testing
- Event-related potential (ERP) experiments
- Motor imagery training with feedback
- Interactive neurofeedback applications
- Behavioral response recording during experiments
- Real-time experimental control and annotation

Technical details:
- Router combines 8 signal channels + 1 event channel = 9 total channels
- Keyboard events appear on channel 8 (last channel)
- Event codes: Up=38, Right=39, Down=40, Left=37 (standard key codes)
- Markers provide visual feedback for successful key presses
- Real-time synchronization between signals and events

Usage:
    python example_basic_keyboard.py
    Press arrow keys to create colored event markers
    Close window to stop the application

Note:
    This example forms the basis for more complex interactive BCI paradigms
    where user input needs to be synchronized with neural signal recording.
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
    router = gp.Router(input_selector=[gp.Router.ALL, gp.Router.ALL])

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
