# --------------------------------------------------------------
# Example file s4e5_choosing_a_file_format.py
# For details and usage, see g.Pype Training Season 4, Episode 5
# --------------------------------------------------------------
#
# Season 4 Episode 1 recorded to CSV, because CSV is the format you can
# open with anything. It is not always the one you want.
#
# g.Pype writes four: CSV, EDF, HDF5 and MATLAB. They are not four ways
# of saying the same thing -- they differ in what metadata survives, how
# large the file is, and which program opens it without an import step.
# This episode records the same signal into all four at once and then
# compares them.

import os
import tempfile
import time

import gpype as gp

SAMPLING_RATE = 250
SECONDS = 4.0
ELECTRODES = ["Fz", "C3", "Cz", "C4"]
OUT = tempfile.gettempdir()


def written(stem: str, extension: str) -> str:
    """The file a writer actually produced, timestamp and all."""
    import glob

    found = sorted(glob.glob(os.path.join(OUT, stem + "_*" + extension)))
    return found[-1] if found else ""


if __name__ == "__main__":

    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=SAMPLING_RATE,
                          channel_count=len(ELECTRODES),
                          signal_frequency=10, signal_amplitude=50.0,
                          noise_amplitude=5.0)

    # Name the channels first, so there is metadata for the formats to
    # differ about. Without this every format writes Ch01..Ch04.
    labeler = gp.ChannelLabeler(labels=ELECTRODES)
    p.connect(source, labeler)

    writers = {
        ".csv": gp.CsvWriter(file_name=os.path.join(OUT, "s4e5.csv")),
        ".edf": gp.EDFWriter(file_name=os.path.join(OUT, "s4e5.edf")),
        ".h5": gp.HDF5Writer(file_name=os.path.join(OUT, "s4e5.h5")),
        ".mat": gp.MatWriter(file_name=os.path.join(OUT, "s4e5.mat")),
    }
    for writer in writers.values():
        p.connect(labeler, writer)

    p.start()
    time.sleep(SECONDS)
    p.stop()
    p.close()

    print()
    print(f"{SECONDS:.0f} s of {len(ELECTRODES)} channels at "
          f"{SAMPLING_RATE} Hz, written four ways:")
    print()
    for extension in (".csv", ".edf", ".h5", ".mat"):
        path = written("s4e5", extension)
        size = os.path.getsize(path) if path else 0
        print(f"  {extension:<6} {size:>9,d} bytes   "
              f"{os.path.basename(path) or '(not written)'}")

    print()
    print("Read those sizes as a trade, not a ranking.")
    print()
    print("CSV is text. Every value is written out in decimal, which is")
    print("why it is the largest, and it is the only one of the four")
    print("you can open in a text editor or a spreadsheet with nothing")
    print("installed. Use it for a short recording someone else has to")
    print("look at without your toolchain.")
    print()
    print("EDF is the clinical interchange format, and the one every")
    print("EEG viewer already reads. It stores samples as 16-bit")
    print("integers scaled between a physical minimum and maximum,")
    print("which is where physical_min and physical_max come in: they")
    print("are the range the integers map onto, so a signal outside")
    print("them is clipped. That quantisation is the price of the")
    print("compatibility.")
    print()
    print("HDF5 keeps the array as it was, in the pipeline's own data")
    print("type, and is the one to reach for when the recording is")
    print("large or when you want to read part of it without loading")
    print("all of it. Python, MATLAB and R all read it.")
    print()
    print("MATLAB (.mat) is HDF5 underneath, written so that load()")
    print("hands the array back under variable_name. Use it when the")
    print("analysis is already in MATLAB, and skip the export step.")
    print()
    print("If you saw a UserWarning from pyedflib about an invalid")
    print("character, that is expected and not your doing: g.Pype")
    print("stamps the non-commercial licence mark into the EDF")
    print("recording-additional header field, and the EDF spec allows")
    print("no spaces there. The mark is written and survives; the")
    print("warning is cosmetic.")
    print()
    print("One thing they all share, and it catches people out: each")
    print("writer appends a timestamp to the name you give it, so the")
    print("file is s4e5_<date>_<time>.csv rather than s4e5.csv. Two")
    print("runs never overwrite each other, and a script cannot find")
    print("its own recording by the path it passed in.")
    print()
    print("Files left in:", OUT)
