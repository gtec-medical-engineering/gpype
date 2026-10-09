"""Season 5, Episode 2: an epoch cut around every key press, and averaged."""

import gpype as gp

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    source = gp.Generator(sampling_rate=250, channel_count=8,
                          signal_frequency=10, signal_amplitude=10,
                          noise_amplitude=10)
    keyboard = gp.Keyboard()

    # One epoch per Up (38) or Right (39) press, from 0.2 s before it to
    # 0.7 s after; any other key is ignored
    trigger = gp.Trigger(time_pre=0.2, time_post=0.7, target=[38, 39])

    # The average of every epoch so far, beside the continuous signal
    epochs = gp.TriggerScope(amplitude_limit=30)
    scope = gp.TimeSeriesScope(amplitude_limit=30, time_window=10)

    p.connect(source, trigger)
    p.connect(keyboard, trigger[gp.Trigger.PORT_TRIGGER])
    p.connect(trigger, epochs)
    p.connect(source, scope)
    app.add_widget(epochs)
    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()
