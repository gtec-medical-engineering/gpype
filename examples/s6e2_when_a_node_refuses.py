# --------------------------------------------------------------
# Example file s6e2_when_a_node_refuses.py
# For details and usage, see g.Pype Training Season 6, Episode 2
# --------------------------------------------------------------
#
# A node checks what its input ports carry before it runs. If two
# streams cannot be combined, it says so and refuses rather than
# producing something meaningless.
#
# This episode triggers the refusal you are most likely to meet -- two
# sources at different sampling rates -- reads what it tells you, and
# then fixes it.

import gpype as gp

if __name__ == "__main__":

    # Two sources that do not agree. One step of a node consumes one
    # frame from *every* input at once, so the inputs cannot run at
    # different rates.
    fast = gp.Generator(sampling_rate=500, channel_count=2)
    slow = gp.Generator(sampling_rate=250, channel_count=2)

    print("--- joining 500 Hz and 250 Hz")
    try:
        p = gp.Pipeline()
        router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])
        p.connect(fast, router["in1"])
        p.connect(slow, router["in2"])
        p.connect(router, gp.Collector())
        p.start()
    except Exception as error:
        print(type(error).__name__ + ":", error)
    finally:
        p.stop()
        p.close()

    # The message names the node, both ports and both rates, and says
    # what to do. So do that: decimate the fast branch down to 250 Hz.
    print()
    print("--- the same pipeline, with a Decimator on the fast branch")
    p = gp.Pipeline()
    fast = gp.Generator(sampling_rate=500, channel_count=2)
    slow = gp.Generator(sampling_rate=250, channel_count=2)
    router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])
    halve = gp.Decimator(decimation_factor=2)
    p.connect(fast, halve)
    p.connect(halve, router["in1"])
    p.connect(slow, router["in2"])
    p.connect(router, gp.Collector())
    print("built without complaint")
    p.close()
