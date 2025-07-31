"""
Basic UDP Send Example - Network Data Transmission via UDP Protocol

This example demonstrates how to stream data over IP networks using the UDP
(User Datagram Protocol) for real-time BCI data transmission. UDP provides
low-latency, connectionless communication ideal for streaming applications
where speed is more important than guaranteed delivery.

What this example shows:
- Generating synthetic 8-channel EEG-like signals with noise
- Capturing keyboard events as experimental markers
- Combining signal and event data using Router
- Streaming combined data over UDP network protocol
- Headless operation (no GUI) for dedicated streaming servers

Expected behavior:
When you run this example:
- UDP packets are sent to the configured network destination
- Data includes 8 signal channels + 1 event channel (9 total)
- Keyboard events are transmitted as numerical markers
- Console shows "Pipeline is running. Press enter to stop."
- Network clients can receive the UDP stream for analysis

Network streaming details:
- Protocol: UDP (User Datagram Protocol)
- Port: Configurable in UDPSender
- Data format: Binary packed multi-channel samples
- Packet rate: 250 Hz (one packet per sample frame)
- Destination: Broadcast or specific IP address

Real-world applications:
- Low-latency BCI control systems
- Real-time signal monitoring across networks
- Distributed processing (send to analysis computers)
- Integration with custom analysis software
- Multi-computer BCI setups
- Remote data logging and backup systems

UDP vs other protocols:
- UDP: Fast, low-latency, no connection overhead (used here)
- TCP: Reliable but higher latency (use for file transfer)
- LSL: Specialized for neuroscience (use for research integration)
- WebSockets: Browser-based applications

Network configuration:
- Sender: This g.Pype application (data source)
- Receiver: Custom application or example_basic_udp_receive.py
- Firewall: May need to allow UDP traffic on specified port
- Local network: Works on same subnet by default

Usage:
    1. Configure network settings in UDPSender if needed
    2. Run: python example_basic_udp_send.py
    3. Use example_basic_udp_receive.py or custom client to receive
    4. Press arrow keys to send event markers over network
    5. Press Enter in console to stop streaming

Note:
    UDP is ideal for real-time applications where occasional packet loss
    is acceptable in exchange for minimal latency and overhead.
"""
import gpype as gp

fs = 250  # Sampling frequency in Hz

if __name__ == "__main__":
    # Create processing pipeline (no GUI needed for UDP streaming)
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
    router = gp.Router(input_selector=[gp.Router.ALL, gp.Router.ALL])

    # UDP sender for low-latency network streaming
    sender = gp.UDPSender()  # Streams to configured UDP destination

    # Connect processing chain: signals + events -> UDP network stream
    p.connect(source, router["in1"])  # Signal data -> Router input 1
    p.connect(keyboard, router["in2"])  # Event data -> Router input 2
    p.connect(router, sender)  # Combined data -> UDP stream

    # Start headless UDP streaming operation
    p.start()  # Begin UDP data transmission
    input("Pipeline is running. Press enter to stop.")  # Wait for user
    p.stop()  # Stop streaming and cleanup
