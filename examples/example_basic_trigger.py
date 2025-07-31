"""
Basic Trigger Example - Event-Related Potential (ERP) Analysis

This example demonstrates how to use the Trigger node to extract time-locked
signal segments around specific events. This is fundamental for analyzing
Event-Related Potentials (ERPs) in BCI applications, where brain responses
to stimuli or user actions are studied.

What this example shows:
- Generating continuous EEG-like signals with background noise
- Capturing keyboard events as experimental triggers
- Extracting signal epochs around trigger events (time-locked segments)
- Dual visualization: continuous signal + triggered epochs
- Real-time ERP analysis capabilities

Expected output:
When you run this example, you'll see two displays:
1. Time Series Scope: Continuous signal with real-time updates
2. Trigger Scope: Event-locked signal epochs overlaid for comparison

Interactive behavior:
- Press ↑ (Up) or → (Right) arrow keys to trigger epoch extraction
- Each key press extracts a 0.9-second epoch (0.2s pre + 0.7s post event)
- Epochs are averaged in the Trigger Scope
- Continuous signal keeps running in the Time Series Scope

Epoch configuration:
- Pre-trigger: 0.2 seconds (baseline period before event)
- Post-trigger: 0.7 seconds (response period after event)
- Target events: Up arrow (38) and Right arrow (39) key codes
- Total epoch length: 0.9 seconds per trigger

Real-world applications:
- P300 speller analysis (visual stimulus responses)
- Auditory ERP studies (sound stimulus processing)
- Visual evoked potential analysis (image/pattern responses)

ERP analysis concepts:
- Time-locking: Aligning signals to specific event times
- Baseline correction: Using pre-trigger period for normalization
- Epoch averaging: Overlaying multiple trials for pattern detection
- Event-related changes: Identifying stimulus-response relationships

Technical details:
- Trigger node monitors event stream for target codes
- Automatic epoch extraction when target events detected
- Signal buffering ensures complete epoch capture
- Real-time display updates for immediate feedback

Usage:
    python example_basic_trigger.py
    Press Up or Right arrow keys to generate triggered epochs
    Observe epoch overlays in the Trigger Scope window

Note:
    This example forms the foundation for more advanced ERP analysis
    and BCI classification systems that rely on event-related signals.
"""
import gpype as gp

fs = 250  # Sampling frequency in Hz

if __name__ == "__main__":
    # Create the main application window
    app = gp.MainApp()

    # Create processing pipeline
    p = gp.Pipeline()

    # Generate continuous EEG-like signals with noise
    source = gp.Generator(
        sampling_rate=fs,
        channel_count=8,  # 8 EEG channels
        signal_frequency=10,  # 10 Hz alpha rhythm
        signal_amplitude=10,  # Signal strength
        signal_shape="sine",  # Clean sine waves
        noise_amplitude=10,
    )  # Background noise

    # Capture keyboard input as trigger events
    keyboard = gp.Keyboard()  # Arrow keys -> trigger codes

    # Trigger node for epoch extraction around events
    trigger = gp.Trigger(
        time_pre=0.2,  # 0.2s before trigger (baseline)
        time_post=0.7,  # 0.7s after trigger (response)
        target=[38, 39],
    )  # Up (38) and Right (39) keys

    # Specialized scope for displaying triggered epochs
    ep_scope = gp.TriggerScope(amplitude_limit=30)  # Overlaid epoch display

    # Standard scope for continuous signal monitoring
    ts_scope = gp.TimeSeriesScope(
        amplitude_limit=30, time_window=10  # Y-axis range
    )  # 10 seconds history

    # Connect the trigger analysis pipeline
    p.connect(source, trigger[gp.Constants.Defaults.PORT_IN])
    p.connect(keyboard, trigger[trigger.PORT_TRIGGER])
    p.connect(trigger, ep_scope)

    # Connect continuous monitoring pipeline
    p.connect(source, ts_scope)  # Continuous signal -> Time Series Scope

    # Add both visualization widgets to application
    app.add_widget(ep_scope)  # Epoch overlay display
    app.add_widget(ts_scope)  # Continuous signal display

    # Start ERP analysis system
    p.start()  # Begin processing (press Up/Right for epochs)
    app.run()  # Show GUI and start main loop
    p.stop()  # Clean shutdown when window closes
