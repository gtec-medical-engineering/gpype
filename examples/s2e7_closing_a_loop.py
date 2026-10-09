"""Season 2, Episode 7: a Delay shifts a stream, or closes a loop."""

import textwrap
import time

import numpy as np

import gpype as gp

SAMPLING_RATE = 250
SECONDS = 1.0
LAG = 25  # samples: a tenth of a second at 250 Hz


def samples(collector):
    """The one channel a Collector kept, as float64."""
    return np.asarray(collector.result.data, dtype=np.float64)[:, 0]


if __name__ == "__main__":

    # 1. Delay a stream: every sample arrives LAG later, the first LAG are 0
    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                          signal_frequency=2, signal_amplitude=1.0)
    delay = gp.Delay(num_samples=LAG)
    original = gp.Collector(name="original")
    delayed = gp.Collector(name="delayed")
    p.connect(source, original)
    p.connect(source, delay)
    p.connect(delay, delayed)
    p.start()
    time.sleep(SECONDS)
    p.stop()
    p.close()

    x, y = samples(original), samples(delayed)
    n = min(len(x), len(y))
    print(f"Delay(num_samples={LAG})")
    print(f"  first {LAG} delayed samples are zero: "
          f"{bool(np.all(y[:LAG] == 0))}")
    print(f"  largest |delayed[n] - original[n - {LAG}]|: "
          f"{np.max(np.abs(y[LAG:n] - x[:n - LAG])):g}")

    # 2. Try to close a loop with it: y[n] = x[n] + 0.5 * y[n - 1]. The
    # plain Delay takes its context from its input, and its input is the
    # loop's own output, so the pipeline refuses the closing connection
    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                          frame_size=1, signal_frequency=0.1,
                          signal_shape="rect", signal_amplitude=1.0)
    feedback = gp.Equation("x + 0.5*y", name="Feedback")
    plain = gp.Delay(num_samples=1, name="Plain")
    p.connect(source, feedback["x"])
    p.connect(feedback, plain)
    print()
    print("closing the loop with Delay(num_samples=1)")
    try:
        p.connect(plain, feedback["y"])
    except ValueError as refusal:
        print(textwrap.fill(f"ValueError: {refusal}", width=72,
                            initial_indent="  ", subsequent_indent="  "))
    p.close()

    # 3. Close it with the declaring form. Given channel_count and
    # sampling_rate, the Delay pushes one frame of initial_value before
    # any input arrives (the y[-1] the first sample needs) and reports
    # BREAKS_CYCLES, which is what the pipeline checks for
    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE, channel_count=1,
                          frame_size=1, signal_frequency=0.1,
                          signal_shape="rect", signal_amplitude=1.0)
    feedback = gp.Equation("x + 0.5*y", name="Feedback")
    breaker = gp.Delay(num_samples=1, channel_count=1,
                       sampling_rate=SAMPLING_RATE, name="Breaker")
    output = gp.Collector(name="output")
    p.connect(source, feedback["x"])
    p.connect(feedback, breaker)
    p.connect(breaker, feedback["y"])
    p.connect(feedback, output)
    print()
    print("closing it with Delay(num_samples=1, channel_count=1, "
          f"sampling_rate={SAMPLING_RATE})")
    print(f"  BREAKS_CYCLES: plain {plain.BREAKS_CYCLES}, "
          f"declaring {breaker.BREAKS_CYCLES}")
    p.start()
    time.sleep(SECONDS)
    p.stop()
    p.close()

    y = samples(output)
    print(f"  first samples: {np.round(y[:6], 4).tolist()}")
    print(f"  after {SECONDS:g} s: {y[-1]:.4f}")
