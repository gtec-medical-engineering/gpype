"""Plot the newest recording from example_basic_csv_writer_record.py.

Globs example_writer*.csv in the current directory, so run the recorder
from here first. Channels are stacked with a fixed vertical offset; the
last one carries the keyboard markers.

Requires: pandas and matplotlib -- neither is a g.Pype dependency
Run: python example_basic_csv_writer_plot.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Find the most recent CSV file from g.Pype recordings
csv_files = glob.glob("example_writer*.csv")
if not csv_files:
    raise FileNotFoundError("No CSV files starting with 'example_' found.")
file_path = max(csv_files, key=os.path.getmtime)  # Most recent file

# Load recorded data into pandas DataFrame. CsvWriter prefixes the file
# with '# NON-COMMERCIAL USE' on an unlicensed install, and with a
# '# device: ...' line when one was recorded, so the comment lines have
# to be skipped or pandas reads the first of them as the header.
data = pd.read_csv(file_path, comment="#")

# Extract time index and channel data
time = data["Time"]  # Sample timestamps
channels = data.columns[1:]  # All data columns (signals + events)

# Create multi-channel EEG-style plot
plt.figure(figsize=(10, 6))

# Channel stacking parameters for clear visualization
offset = -100  # Vertical spacing between channels
yticks = []  # Y-axis tick positions
yticklabels = []  # Y-axis tick labels

# Plot each channel with vertical offset
for i, ch in enumerate(channels):
    channel_offset = i * offset
    plt.plot(time, data[ch] + channel_offset, label=ch)
    yticks.append(channel_offset)
    yticklabels.append(f"Ch{i + 1}")

# Configure plot appearance
plt.yticks(yticks, yticklabels)
plt.xlabel("Time (s)")
plt.title("EEG Recordings")
plt.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.ylim((len(channels)) * offset, -offset)

# Display the plot
plt.tight_layout()
plt.show()
