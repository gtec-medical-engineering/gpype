"""
Basic Router Example - Channel Selection and Data Stream Routing

This example demonstrates how to use the Router node to selectively combine
channels from multiple data sources into a single output stream. The Router
is essential for flexible data management in complex BCI pipelines where
different signal sources need to be combined or specific channels selected.

What this example shows:
- Creating two different signal generators (sine and rectangular waves)
- Using Router to select specific channels from each source
- Combining selected channels into a unified output stream
- Flexible channel mapping and data stream management

Expected output:
When you run this example, you'll see:
- 5 total channels in the scope (2 from sine + 3 from rectangular)
- Channels 1-2: 10 Hz sine waves from first generator
- Channels 3-5: 2 Hz rectangular waves from second generator
- Clear visual distinction between the two signal types
- Demonstrates how Router preserves signal characteristics while combining

Router configuration:
- Input 1: Channels 0,1 from sine wave generator (-> output channels 1-2)
- Input 2: Channels 3,4,5 from square-wave generator (-> output channels 3-5)
- Total output: 5 channels with mixed signal types

Real-world applications:
- Multi-device integration (combining EEG + EMG + EOG signals)
- Channel subset selection (choosing specific electrode locations)
- Signal source switching (alternating between different input sources)
- Data stream merging (combining multiple acquisition systems)
- Feature channel selection (using only relevant channels for analysis)

Advanced routing scenarios:
- Spatial filtering (selecting channels by brain region)
- Artifact channel isolation (separating signal from noise channels)
- Multi-modal BCI (combining different signal modalities)
- Dynamic channel reconfiguration during experiments
- Cross-validation with different channel subsets

Technical details:
- Router automatically handles different sampling rates (if matched)
- Channel indexing starts from 0 in input, 1 in display
- Input channels can be reordered or duplicated in output
- Router preserves data types and timing information
- Multiple Router nodes can be chained for complex routing

Usage:
    python example_basic_router.py

Note:
    The Router is one of the most versatile nodes in g.Pype, enabling
    sophisticated data flow management essential for complex BCI systems.
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
    router = gp.Router(input_selector=[[0, 1], [3, 4, 5]])

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
