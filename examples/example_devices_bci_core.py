"""
BCI Core-4/8 - real-time EEG acquisition from either device

``gp.BCICore()`` drives the whole family: it takes the first amplifier
found and adopts the rate it reports. The channel count is not
discovered -- it defaults to 8, so a Core-4 needs channel_count=4 or a
four-label montage. Only 8 channels get default electrode labels
(Fz C3 Cz C4 Pz PO7 POz PO8); name any other montage yourself.

Requires: a BCI Core-4 or Core-8, and pip install gpype[all]
Run: python example_devices_bci_core.py
"""
import gpype as gp

# Electrode names, or None to accept the device's own.
#
# None is right on a Core-8: it ships a fixed headset and g.Pype knows
# its labels. On a Core-4 there is no default, so name the four
# electrodes you actually mounted -- these are a placeholder.
MONTAGE = None
# MONTAGE = gp.Montage(["Fz", "Cz", "Pz", "Oz"])   # e.g. a Core-4

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    # === HARDWARE DATA SOURCE ===
    # No serial, no channel count, no sampling rate: the first amplifier
    # discovered is used, and it reports its own configuration. Pass
    # `serial=` to pick a specific device when more than one is in range,
    # or `channel_count=` to take fewer channels than it offers.
    source = gp.BCICore(montage=MONTAGE)

    # === SIGNAL CONDITIONING ===
    # 1-30 Hz keeps the major brain rhythms and drops DC drift below and
    # muscle artefact above.
    bandpass = gp.Bandpass(f_lo=1, f_hi=30)

    # Mains interference, both standards, so the example travels.
    notch50 = gp.Bandstop(f_lo=48, f_hi=52)
    notch60 = gp.Bandstop(f_lo=58, f_hi=62)

    # === VISUALIZATION ===
    # +/-50 uV covers typical scalp EEG; 10 s gives useful context.
    scope = gp.TimeSeriesScope(amplitude_limit=50, time_window=10)

    # === PIPELINE ===
    p.connect(source, bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50, notch60)
    p.connect(notch60, scope)

    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
