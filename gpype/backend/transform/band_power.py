from __future__ import annotations

from typing import Optional

import numpy as np

from ...common._private import channels
from ...common.constants import Constants
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT

#: The conventional EEG bands, in Hz.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


class BandPower(IONode):
    """Reduces a spectrum to power in named frequency bands.

    This is the most common feature in online BCI work, and until now
    nothing could consume an FFT at all: the spectrum carried no
    frequency axis, so a scope was its only possible destination.

    Bands may be named -- the conventional EEG bands are built in -- or
    given as explicit ranges. The output is one value per band per
    channel, labelled so that a downstream file or stream says which is
    which rather than leaving the reader to count columns.
    """

    #: The conventional EEG bands, by name -- which is the form an
    #: author writes. Not the resolved [name, low, high] triples: those
    #: are the internal shape, and publishing them as the default told
    #: an authoring tool that a triple is what a person has to type.
    DEFAULT_BANDS: tuple = tuple(BANDS)

    class Configuration(IONode.Configuration):
        """Configuration class for BandPower parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for BandPower settings."""

            #: Bands to measure, as name or [low, high] pairs
            BANDS = "bands"

    def __init__(self, bands=None, **kwargs):
        """Initialize the band power node.

        Args:
            bands: Band names from BANDS, or explicit ``[low, high]``
                pairs in Hz, or a mix. Defaults to the conventional EEG
                bands.
            **kwargs: Additional arguments for the parent IONode.

        Raises:
            ValueError: If a band name is unknown, a range is malformed,
                or the list is empty.
        """
        if bands is None:
            bands = list(self.DEFAULT_BANDS)
        if isinstance(bands, str):
            bands = [bands]
        if len(bands) == 0:
            raise ValueError("bands must not be empty.")

        resolved = []
        for band in bands:
            if isinstance(band, str):
                if band not in BANDS:
                    raise ValueError(
                        f"Unknown band '{band}'. Known bands: "
                        f"{sorted(BANDS)}"
                    )
                low, high = BANDS[band]
                resolved.append([band, float(low), float(high)])
            elif len(band) == 3 and isinstance(band[0], str):
                # Already resolved, which is the form stored in the
                # configuration: a rebuilt node must accept what the
                # original wrote.
                resolved.append([band[0], float(band[1]), float(band[2])])
            else:
                if len(band) != 2:
                    raise ValueError(
                        f"A band range needs a low and a high value, got "
                        f"{band}."
                    )
                low, high = float(band[0]), float(band[1])
                if low >= high:
                    raise ValueError(
                        f"Band low must be below high, got {band}."
                    )
                resolved.append([f"{low:g}-{high:g}Hz", low, high])

        super().__init__(bands=resolved, **kwargs)
        self._masks = None  # One boolean bin mask per band

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Rebuild the frequency axis and resolve each band to bins.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts.

        Raises:
            ValueError: If the input is not a spectrum, or a band falls
                entirely outside the represented frequencies.
        """
        port_context_out = super().setup(data, port_context_in)
        context = port_context_in[PORT_IN]

        rate = context.get(Constants.Keys.SAMPLING_RATE)
        bins = context.get(Constants.Keys.FRAME_SIZE)
        if not rate or not bins or bins < 2:
            raise ValueError(
                "BandPower needs a spectrum: connect it downstream of an "
                "FFT, whose output context carries the sampling rate and "
                "the number of frequency bins."
            )

        # An FFT emits the one-sided spectrum of a window, so the window
        # length follows from the bin count and gives back the axis.
        window = (bins - 1) * 2
        freqs = np.fft.rfftfreq(window, 1.0 / rate)

        bands = self.config[self.Configuration.Keys.BANDS]
        self._masks = []
        for name, low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            if not mask.any():
                raise ValueError(
                    f"Band '{name}' ({low}-{high} Hz) contains no "
                    f"frequency bins; the spectrum spans "
                    f"{freqs[0]:g}-{freqs[-1]:g} Hz."
                )
            self._masks.append(mask)

        channel_count = channels.channel_count(context)
        names = (
            list(channels.labels_of(context))
            if channels.has_labels(context)
            else [f"Ch{i + 1:02d}" for i in range(channel_count)]
        )
        labels = [f"{ch}:{band[0]}" for ch in names for band in bands]

        out = port_context_out[PORT_OUT]
        out[Constants.Keys.CHANNEL_COUNT] = len(labels)
        out[Constants.Keys.FRAME_SIZE] = 1
        out.update(
            channels.describe(
                [Constants.ChannelRoles.SIGNAL] * len(labels), labels, None
            )
        )
        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Sum the spectrum over each band.

        Args:
            data: One spectrum, bins by channels.

        Returns:
            One row, channel-major with one value per band.
        """
        spectrum = data[PORT_IN]
        values = [spectrum[mask, :].sum(axis=0) for mask in self._masks]
        # Channel-major so a channel's bands stay adjacent, matching the
        # labels published at setup.
        stacked = np.stack(values, axis=1)
        return {PORT_OUT: stacked.reshape(1, -1)}
