"""Season 8, Episode 1: one script, run as the edge and as the server.

Start the server first, then the edge, each in its own terminal:

  python s8e1_one_script_two_processes.py --residency server
      --endpoint ws://localhost:8765 --bind loopback
  python s8e1_one_script_two_processes.py --residency edge
      --endpoint ws://localhost:8765

The server half is a deployment: it needs a g.Pype Runtime licence.
"""

import gpype as gp

if __name__ == "__main__":

    # --residency, --endpoint and --bind, read from the command line.
    config = gp.LaunchConfig.parse_args()

    p = gp.Pipeline()
    # Both processes build this pipeline, and the two halves of a Link
    # find each other by the node's id. A node built in code gets a new
    # id in every process, so give the source one.
    source = gp.Generator(sampling_rate=250, channel_count=4,
                          signal_frequency=10, signal_amplitude=20.0,
                          id="s8e1-eeg")
    band = gp.Bandpass(f_lo=8, f_hi=12)
    out = gp.Collector()
    p.connect(source, band)
    p.connect(band, out)

    # What this process built for the source: the pipeline's answer.
    built = [type(n).__name__ for n in p.chain_of(source).internal_nodes]
    print(f"{config.residency}: the source is {built}")

    p.start()
    input(f"{config.residency}: running. Press Enter to stop.\n")
    p.stop()

    # A Collector touches no device, file or screen, so every process
    # builds one. Only the server's is fed: the edge's stream leaves
    # through its Link.
    result = out.result
    got = "nothing" if result is None else f"{result.sample_count} samples"
    print(f"{config.residency}: collected {got}")
    p.close()
