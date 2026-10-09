"""Season 4, Episode 3: signal and key presses sent as UDP packets.

Headless on purpose: the pipeline streams until you press Enter. Read
it with s4e3_foreign_receiver.py. UDP drops a packet rather than delay
it, which is the trade this episode makes.
"""

import gpype as gp

if __name__ == "__main__":

    p = gp.Pipeline()

    source = gp.Generator(sampling_rate=250, channel_count=8,
                          signal_frequency=10, signal_amplitude=10,
                          noise_amplitude=10)
    keyboard = gp.Keyboard()
    router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])

    # One packet per frame, to 127.0.0.1 port 56000 unless you pass ip
    # and port; frame_size_out would gather samples into larger packets
    sender = gp.UDPSender()

    p.connect(source, router["in1"])
    p.connect(keyboard, router["in2"])
    p.connect(router, sender)

    p.start()
    input("Streaming. Press Enter to stop.")
    p.stop()
    p.close()
