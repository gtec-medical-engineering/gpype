"""Season 4, Episode 11: every source recorded at launch, and replayed.

Nothing in the pipeline changes: --save-as writes a run directory with
one HDF5 file per source, named after the node, and --load-from replays
it with no keyboard present. The two flags together are refused.
"""

from pathlib import Path

import gpype as gp

if __name__ == "__main__":

    # What was asked for on the command line, read only to say so; the
    # pipeline below does not consult it
    launch = gp.LaunchConfig.get()
    if launch.load_from:
        print(f"Replaying '{launch.load_from}'. Hands off the keyboard.")
    elif launch.save_as:
        print(f"Recording every source under '{launch.save_as}_*'.")
    else:
        print("Live run, nothing recorded. To record, pass --save-as "
              "my_run; to replay, pass --load-from my_run_<timestamp>")

    app = gp.MainApp()
    p = gp.Pipeline()

    # The names are what the recordings are filed under: eeg.h5, keys.h5
    source = gp.Generator(name="eeg", sampling_rate=250, channel_count=8,
                          signal_frequency=10, signal_amplitude=10,
                          noise_amplitude=10)
    keyboard = gp.Keyboard(name="keys")
    router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])

    # A replayed press shows as the same marker, on the same sample
    mk = gp.TimeSeriesScope.Markers
    markers = [mk(color="r", label="up", channel=8, value=38),
               mk(color="g", label="right", channel=8, value=39),
               mk(color="c", label="down", channel=8, value=40),
               mk(color="y", label="left", channel=8, value=37)]
    scope = gp.TimeSeriesScope(amplitude_limit=30, time_window=10,
                               markers=markers, hidden_channels=[8])

    p.connect(source, router["in1"])
    p.connect(keyboard, router["in2"])
    p.connect(router, scope)
    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()

    # The run directory carries a timestamp, so it is looked up rather
    # than guessed
    if launch.save_as:
        stem = Path(launch.save_as)
        runs = sorted(stem.parent.glob(stem.name + "_*"))
        if runs:
            newest = runs[-1]
            print(f"\nRecorded {len(list(newest.glob('*.h5')))} stream(s):")
            for entry in sorted(newest.iterdir()):
                print(f"  {entry.name}")
            print(f"\nReplay it with:\n"
                  f"  python {Path(__file__).name} --load-from {newest}")
