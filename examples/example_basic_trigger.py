"""Extracting signal epochs time-locked to keyboard events.

Each Up (38) or Right (39) key press emits one epoch spanning 0.2 s
before to 0.7 s after it. TriggerScope overlays the epochs; the
TimeSeriesScope keeps showing the continuous signal.

Requires: gpype[gui,devices]
Run: python example_basic_trigger.py
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
