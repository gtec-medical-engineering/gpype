"""Season 4, Episode 4: trigger codes received over UDP, beside the signal.

The stimulus program is the thread below, and nothing in it imports
gpype: a plain socket sending a number as text. In a real session it
is somebody else's process, often on another machine; delete the
thread and keep the rest.
"""

import socket
import threading

import gpype as gp

# The default port, 1000, needs root on macOS and Linux: use one from
# 1024 to 49151, and point the sender at the same one
PORT = 5005


def stimulus_program(stop: threading.Event) -> None:
    """Send the codes 1 and 2 in turn, one a second, as text."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    code = 1
    while not stop.wait(1.0):
        sock.sendto(str(code).encode(), ("127.0.0.1", PORT))
        code = 3 - code
    sock.close()


if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    source = gp.Generator(sampling_rate=250, channel_count=8,
                          signal_frequency=10, signal_amplitude=10,
                          noise_amplitude=10)

    # Each datagram's number becomes one sample on a trigger channel,
    # followed by a 0 ten milliseconds later
    receiver = gp.UDPReceiver(ip="127.0.0.1", port=PORT)

    # The generator's 8 channels, then the trigger: codes on channel 8
    router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])

    mk = gp.TimeSeriesScope.Markers
    markers = [mk(color="r", label="one", channel=8, value=1),
               mk(color="g", label="two", channel=8, value=2)]
    scope = gp.TimeSeriesScope(amplitude_limit=30, time_window=10,
                               markers=markers, hidden_channels=[8])

    p.connect(source, router["in1"])
    p.connect(receiver, router["in2"])
    p.connect(router, scope)
    app.add_widget(scope)

    stop = threading.Event()
    threading.Thread(target=stimulus_program, args=(stop,),
                     daemon=True).start()

    p.start()
    app.run()
    p.stop()
    p.close()
    stop.set()
