"""Publishing signals and keyboard events as an LSL stream.

Headless on purpose: there is no MainApp, so the pipeline runs until
you press Enter. Read the stream back with example_basic_lsl_receive.py,
or with any other LSL consumer.

Requires: gpype[lsl,devices]
Run: python example_basic_lsl_send.py
"""
import gpype as gp

fs = 250  # Sampling frequency in Hz

if __name__ == "__main__":
    # Create processing pipeline (no GUI needed for streaming)
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
    keyboard = gp.Keyboard()  # Arrow keys -> event codes

    # Combine signal data (8 channels) + keyboard events (1 channel)
    router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])

    # LSL sender for network streaming
    sender = gp.LSLSender()  # Creates discoverable LSL stream

    # Connect processing chain: signals + events -> network stream
    p.connect(source, router["in1"])  # Signal data -> Router input 1
    p.connect(keyboard, router["in2"])  # Event data -> Router input 2
    p.connect(router, sender)  # Combined data -> LSL stream

    # Start headless streaming operation
    p.start()  # Begin data streaming
    input("Pipeline is running. Press enter to stop.")  # Wait for user
    p.stop()  # Stop streaming and cleanup
