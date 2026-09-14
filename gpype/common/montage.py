from __future__ import annotations

from typing import Iterator, Optional

#: Normalised 2-D scalp positions for the international 10-20 system.
#: x runs left(-1) to right(+1), y runs posterior(-1) to anterior(+1),
#: both on the unit circle of the projected head outline. Used for
#: topographic layout; not intended for source localisation.
_STANDARD_1020: dict[str, tuple[float, float]] = {
    "Fp1": (-0.31, 0.95),
    "Fpz": (0.00, 1.00),
    "Fp2": (0.31, 0.95),
    "F7": (-0.81, 0.59),
    "F3": (-0.45, 0.63),
    "Fz": (0.00, 0.50),
    "F4": (0.45, 0.63),
    "F8": (0.81, 0.59),
    "T7": (-1.00, 0.00),
    "C3": (-0.50, 0.00),
    "Cz": (0.00, 0.00),
    "C4": (0.50, 0.00),
    "T8": (1.00, 0.00),
    "P7": (-0.81, -0.59),
    "P3": (-0.45, -0.63),
    "Pz": (0.00, -0.50),
    "P4": (0.45, -0.63),
    "P8": (0.81, -0.59),
    "O1": (-0.31, -0.95),
    "Oz": (0.00, -1.00),
    "O2": (0.31, -0.95),
    "A1": (-1.10, 0.00),
    "A2": (1.10, 0.00),
    "M1": (-0.95, -0.30),
    "M2": (0.95, -0.30),
}

#: Historical aliases accepted on input and normalised to 10-20 names.
_ALIASES: dict[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}

#: Identifier for the international 10-20 electrode system.
SYSTEM_1020: str = "10-20"
#: Identifier for a montage whose labels are not from a known system.
SYSTEM_CUSTOM: str = "custom"


class Montage:
    """Ordered electrode labels for the EEG channels of a source.

    A montage names the signal channels of an acquisition device. It
    describes *which electrode sits on which channel*, and is supplied by
    the user because amplifiers do not report electrode names.

    A montage covers the signal channels only. Auxiliary channels a device
    may stream alongside EEG - counter, saturation, battery, validity -
    are labelled by the source itself and are not part of the montage.

    Positions are resolved from a built-in table of standard electrode
    names rather than stored per instance, so a montage stays cheap to
    serialise. Labels outside the table simply have no position.

    Example:
        >>> m = Montage(["Fz", "Cz", "Pz", "C3", "C4", "O1", "O2", "Oz"])
        >>> len(m)
        8
        >>> m.index_of("Cz")
        1
        >>> m.position_of("Cz")
        (0.0, 0.0)
    """

    #: Ordered electrode labels
    _labels: tuple[str, ...]
    #: Electrode used as the hardware reference, if known
    _reference: Optional[str]

    def __init__(
        self,
        labels: list[str],
        reference: Optional[str] = None,
    ):
        """Initialize a montage from an ordered list of electrode labels.

        Args:
            labels: Electrode labels, in channel order. Historical names
                (T3, T4, T5, T6) are normalised to their 10-20
                equivalents.
            reference: Label or description of the hardware reference,
                e.g. "Cz" or "linked_mastoids". Recorded as metadata
                only; re-referencing is a separate processing step.

        Raises:
            ValueError: If labels is empty, contains non-strings, blank
                entries, or duplicates.
        """
        if not labels:
            raise ValueError("montage must contain at least one label.")
        if not all(isinstance(lbl, str) for lbl in labels):
            raise ValueError("all montage labels must be strings.")

        normalised = [self.normalise(lbl) for lbl in labels]
        if any(not lbl for lbl in normalised):
            raise ValueError("montage labels must not be blank.")

        duplicates = {lbl for lbl in normalised if normalised.count(lbl) > 1}
        if duplicates:
            raise ValueError(
                f"duplicate montage labels: {sorted(duplicates)}. Labels "
                f"must be unique so channels can be addressed by name."
            )

        self._labels = tuple(normalised)
        self._reference = self.normalise(reference) if reference else None

    @staticmethod
    def normalise(label: str) -> str:
        """Return the canonical form of an electrode label.

        Args:
            label: Electrode label, possibly a historical alias.

        Returns:
            The canonical label, with surrounding whitespace removed.
        """
        stripped = label.strip()
        return _ALIASES.get(stripped, stripped)

    @classmethod
    def standard_1020(
        cls, labels: list[str], reference: Optional[str] = None
    ) -> "Montage":
        """Create a montage and require every label to be a 10-20 name.

        Use this when a typo should be an error rather than an unnamed
        channel.

        Args:
            labels: Electrode labels, in channel order.
            reference: Hardware reference, if known.

        Returns:
            The validated montage.

        Raises:
            ValueError: If any label is not part of the 10-20 system.
        """
        montage = cls(labels, reference=reference)
        unknown = [lbl for lbl in montage.labels if lbl not in _STANDARD_1020]
        if unknown:
            raise ValueError(
                f"not 10-20 electrode names: {unknown}. Known names: "
                f"{sorted(_STANDARD_1020)}"
            )
        return montage

    @property
    def labels(self) -> tuple[str, ...]:
        """Electrode labels in channel order."""
        return self._labels

    @property
    def reference(self) -> Optional[str]:
        """Hardware reference electrode or description, if known."""
        return self._reference

    @property
    def system(self) -> str:
        """Name of the electrode system, if all labels belong to one."""
        if all(lbl in _STANDARD_1020 for lbl in self._labels):
            return SYSTEM_1020
        return SYSTEM_CUSTOM

    def index_of(self, label: str) -> int:
        """Return the channel index carrying the given electrode.

        Args:
            label: Electrode label, alias accepted.

        Returns:
            Zero-based channel index.

        Raises:
            KeyError: If the electrode is not part of this montage.
        """
        canonical = self.normalise(label)
        try:
            return self._labels.index(canonical)
        except ValueError:
            raise KeyError(
                f"{label!r} is not in this montage: {list(self._labels)}"
            ) from None

    def select(self, labels: list[str]) -> list[int]:
        """Return channel indices for a subset of electrodes, in order.

        Args:
            labels: Electrode labels to look up.

        Returns:
            Channel indices in the order the labels were given.

        Raises:
            KeyError: If any electrode is not part of this montage.
        """
        return [self.index_of(lbl) for lbl in labels]

    def position_of(self, label: str) -> Optional[tuple[float, float]]:
        """Return the normalised 2-D scalp position of an electrode.

        Args:
            label: Electrode label, alias accepted.

        Returns:
            ``(x, y)`` on the projected head outline, or None if the
            label is not a known standard electrode.
        """
        return _STANDARD_1020.get(self.normalise(label))

    def positions(self) -> list[Optional[tuple[float, float]]]:
        """Return positions for all channels, in channel order.

        Returns:
            One entry per channel; None where the label is not a known
            standard electrode.
        """
        return [self.position_of(lbl) for lbl in self._labels]

    def __len__(self) -> int:
        """Number of channels described by this montage."""
        return len(self._labels)

    def __iter__(self) -> Iterator[str]:
        """Iterate over electrode labels in channel order."""
        return iter(self._labels)

    def __getitem__(self, index: int) -> str:
        """Return the electrode label of a channel.

        Args:
            index: Zero-based channel index.

        Returns:
            Electrode label.
        """
        return self._labels[index]

    def __eq__(self, other: object) -> bool:
        """Compare montages by labels and reference."""
        if not isinstance(other, Montage):
            return NotImplemented
        return (
            self._labels == other._labels
            and self._reference == other._reference
        )

    def __repr__(self) -> str:
        """Return a concise representation for logs and errors."""
        ref = f", reference={self._reference!r}" if self._reference else ""
        return f"Montage({list(self._labels)}{ref}, system={self.system!r})"
