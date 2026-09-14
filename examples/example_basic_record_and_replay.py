"""Recording every source's raw data, then replaying it.

Notice what is NOT in this file: no reader node, no writer node, no
branch on live versus replayed -- recording and replay are decided at
launch. --save-as writes a run DIRECTORY holding one HDF5 file per
source, named after the node, so name your nodes; --load-from replays
them with no keyboard or amplifier present. The two flags together
are refused.

Requires: gpype[all]
Run: python example_basic_record_and_replay.py [--save-as my_run]
"""
from pathlib import Path

import gpype as gp

fs = 250  # Sampling frequency in Hz

if __name__ == "__main__":
    # What was asked for on the command line. Read only to explain
    # itself -- the pipeline below does not consult it.
    launch = gp.LaunchConfig.get()
    if launch.load_from:
        print(f"Replaying '{launch.load_from}'. Hands off the keyboard.")
    elif launch.save_as:
        print(f"Recording every source under '{launch.save_as}_*'.")
    else:
        print(
            "Live run; nothing is being recorded. To record it, pass\n"
            "  --save-as my_run\n"
            "and to replay what you recorded, pass\n"
            "  --load-from my_run_<timestamp>"
        )

    # Create the main application window
    app = gp.MainApp()

    # Create processing pipeline
    p = gp.Pipeline()

    # Generate synthetic 8-channel EEG-like signals. The name is what
    # this stream's recording is filed under: eeg.h5.
    source = gp.Generator(
        name="eeg",
        sampling_rate=fs,
        channel_count=8,
        signal_frequency=10,  # 10 Hz alpha-like rhythm
        signal_amplitude=10,
        signal_shape="sine",
        noise_amplitude=10,
    )

    # Capture arrow keys as event markers -> keys.h5
    keyboard = gp.Keyboard(name="keys")

    # Combine 8 signal channels + 1 event channel = 9 channels
    router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])

    # Colour the arrow keys so a replayed keypress is visibly the same
    # event, on the same sample, as the one that was recorded.
    mk = gp.TimeSeriesScope.Markers
    markers = [
        mk(color="r", label="up", channel=8, value=38),
        mk(color="g", label="right", channel=8, value=39),
        mk(color="b", label="down", channel=8, value=40),
        mk(color="k", label="left", channel=8, value=37),
    ]

    scope = gp.TimeSeriesScope(
        amplitude_limit=30,
        time_window=10,
        markers=markers,
    )

    # Connect processing chain
    p.connect(source, router["in1"])  # Signals -> Router input 1
    p.connect(keyboard, router["in2"])  # Events  -> Router input 2
    p.connect(router, scope)  # Combined -> Display

    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()

    # Hand over the command that replays what was just recorded. The run
    # directory carries a timestamp so a second recording never
    # overwrites the first, which is exactly why it has to be looked up
    # rather than guessed.
    if launch.save_as:
        stem = Path(launch.save_as)
        runs = sorted(stem.parent.glob(stem.name + "_*"))
        if runs:
            newest = runs[-1]
            print(f"\nRecorded {len(list(newest.glob('*.h5')))} stream(s):")
            for entry in sorted(newest.iterdir()):
                print(f"  {entry.name}")
            print(
                f"\nReplay it with:\n"
                f"  python {Path(__file__).name} --load-from {newest}"
            )
