"""Season 4, Episode 2: signal and key presses published as an LSL stream.

Headless on purpose: the pipeline streams until you press Enter. Read
it back with s4e2_foreign_consumer.py, or with any other LSL consumer.
"""

import gpype as gp

if __name__ == "__main__":

    p = gp.Pipeline()

    source = gp.Generator(sampling_rate=250, channel_count=8,
                          signal_frequency=10, signal_amplitude=10,
                          noise_amplitude=10)
    keyboard = gp.Keyboard()
    router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])

    # One stream of type EEG, named gpype_lsl unless you pass stream_name
    sender = gp.LSLSender()

    p.connect(source, router["in1"])
    p.connect(keyboard, router["in2"])
    p.connect(router, sender)

    p.start()
    input("Streaming. Press Enter to stop.")
    p.stop()
    p.close()
