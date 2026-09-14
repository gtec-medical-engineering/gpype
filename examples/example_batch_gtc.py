"""Read a `.gtc` recording as one batch block and print what it knows.

A `.gtc` carries what a CSV cannot: the unit of every channel, the
absolute time of the first sample, the markers, where the recording was
interrupted, and whether it was sealed cleanly. All of it arrives on the
Result. Consult `.gaps` before cutting epochs -- samples either side of a
dropout are not adjacent, however they look.

Requires: the `gtc` package, and a recording of at least 60 s
Run: python example_batch_gtc.py path/to/recording.gtc
"""

import sys
from collections import Counter

import gpype as gp


def summarise(result, path: str) -> None:
    """Print what the recording says about itself.

    Args:
        result: The Result a batch run returned.
        path: The file it came from.
    """
    print(path)
    print(
        f"  {result.channel_count} ch x {result.rate:g} Hz, "
        f"{result.duration:.1f} s"
        + (
            f", starting {result.start_time.isoformat()}"
            if result.start_time
            else ""
        )
    )

    # How much the file vouches for itself: an unsealed recording is a
    # clean prefix of a run that was interrupted, and a result derived
    # from one should say so wherever it is shown.
    if result.trust:
        print(f"  trust     : {result.trust}")

    if result.units:
        shown = result.units[:3]
        more = " ..." if len(result.units) > 3 else ""
        print(f"  units     : {shown}{more}")

    if result.gaps:
        print(f"  gaps      : {len(result.gaps)}")
        for gap in result.gaps:
            seconds = gap.n_missing / float(result.rate)
            print(
                f"    {seconds:.1f} s missing from sample "
                f"{gap.first_missing} ({gap.reason})"
            )
    else:
        print("  gaps      : none -- the recording is continuous")

    if result.events:
        counted = Counter(event.label for event in result.events)
        summary = ", ".join(
            f"'{label}' x {count}" for label, count in counted.most_common(4)
        )
        print(f"  events    : {len(result.events)}  ({summary})")
    else:
        print("  events    : none")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(__doc__.split("Usage:")[0].strip())
        print()
        print("Usage: python example_batch_gtc.py path/to/recording.gtc")
        sys.exit(1)

    path = sys.argv[1]

    # The whole recording, everything it knows.
    with gp.Pipeline() as p:
        reader = gp.GtcReader(file_name=path)
        collector = gp.Collector()
        p.connect(reader, collector)
        whole = p.run()

    summarise(whole, path)

    # A subset: four channels of the first minute, zero-phase filtered.
    # Naming channels rather than counting them is the point of the file
    # carrying their names.
    labels = (whole.labels or [])[:4]
    print()
    print(f"Re-reading {labels} for the first minute, zero phase ...")

    with gp.Pipeline() as p:
        reader = gp.GtcReader(
            file_name=path,
            channels=labels,
            start=0,
            stop=int(60 * float(whole.rate)),
        )
        bandpass = gp.Bandpass(f_lo=1, f_hi=40, phase="zero")
        collector = gp.Collector()
        p.connect(reader, bandpass)
        p.connect(bandpass, collector)
        minute = p.run()

    print(minute)
    print(f"  labels : {minute.labels}")

    # Anything that cuts trials has to consult this first.
    if minute.gaps:
        print(
            f"  WARNING: {len(minute.gaps)} interruption(s) inside this "
            f"minute -- do not cut epochs across them."
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nInstall matplotlib to see the plot.")
    else:
        axes = minute.plot()
        axes.set_title(f"{path}: first minute, 1-40 Hz zero phase")
        if minute.units:
            axes.set_ylabel(f"amplitude ({minute.units[0]})")
        axes.legend(loc="upper right", fontsize="small")
        plt.tight_layout()
        plt.show()
