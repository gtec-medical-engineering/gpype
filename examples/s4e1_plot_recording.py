"""Season 4, Episode 1: the newest recording, plotted with pandas.

Neither pandas nor matplotlib is a g.Pype dependency:
pip install pandas matplotlib
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    # The newest file the recorder wrote into this folder
    files = glob.glob("s4e1_*.csv")
    if not files:
        raise FileNotFoundError(f"no s4e1_*.csv in {os.getcwd()}")
    path = max(files, key=os.path.getmtime)

    # The file opens with comment lines, and its header puts a space
    # after each comma. Without these two arguments pandas takes the
    # first comment line as the header, and names a column " Ch01"
    data = pd.read_csv(path, comment="#", skipinitialspace=True)

    time = data["Time"]
    channels = data.columns[1:]

    # One lane per channel, stacked with a fixed offset; the last lane
    # carries the key codes
    offset = -100
    plt.figure(figsize=(10, 6))
    for i, ch in enumerate(channels):
        plt.plot(time, data[ch] + i * offset, label=ch)
    plt.yticks([i * offset for i in range(len(channels))], channels)
    plt.xlabel("Time (s)")
    plt.title(os.path.basename(path))
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.ylim(len(channels) * offset, -offset)
    plt.tight_layout()
    plt.show()
