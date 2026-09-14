from __future__ import annotations

from typing import Optional

import numpy as np

from ...common._private import channels
from ...common.constants import Constants
from ...common.montage import Montage
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class ChannelLabeler(IONode):
    """Says what each channel is, without touching the data.

    Only some sources can name their own channels, and none of them can
    know what a user has wired downstream. Without names a recording
    lands as Ch01..ChNN and a stream goes out unnamed, which is a problem
    on analysis day rather than on recording day.

    It also carries roles, which say how a channel must be treated rather
    than what it is. Marking the extra channels an amplifier appends --
    a digital input as a trigger, an accelerometer as auxiliary, a link
    quality as quality -- is what stops a filter from smearing them.

    The data passes through untouched; only the description changes.
    """

    #: At most one of these -- "Give either a montage or labels, not
    #: both". Distinct from ONE_OF: both may be absent, so this is not a
    #: requirement, only an exclusion.
    AT_MOST_ONE_OF = (("montage", "labels"),)

    class Configuration(IONode.Configuration):
        """Configuration class for ChannelLabeler parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for ChannelLabeler settings."""

            pass

        class OptionalKeys(IONode.Configuration.OptionalKeys):
            """Optional configuration keys."""

            #: Electrode labels, one per channel
            LABELS = "labels"
            #: Per-channel roles, one per channel
            ROLES = "roles"
            #: Electrode system the labels come from
            SYSTEM = "system"

    def __init__(
        self,
        labels: Optional[list] = None,
        roles: Optional[list] = None,
        montage: Optional[Montage] = None,
        **kwargs,
    ):
        """Initialize the channel labeler.

        Args:
            labels: One name per channel, e.g. ``["C3", "Cz", "C4"]``.
            roles: One role per channel, from Constants.ChannelRoles.
                Channels left unnamed here count as measured signal.
            montage: A Montage to take the labels from, as an alternative
                to passing them directly.
            **kwargs: Additional arguments for the parent IONode.

        Raises:
            ValueError: If both a montage and labels are given, if either
                list is empty, or if a role is not a known role.
        """
        if montage is not None:
            if labels is not None:
                raise ValueError("Give either a montage or labels, not both.")
            labels = list(montage.labels)

        if labels is not None:
            if not labels:
                raise ValueError("labels must not be empty.")
            labels = [str(label) for label in labels]
            kwargs.setdefault(self.Configuration.OptionalKeys.LABELS, labels)
        if roles is not None:
            if not roles:
                raise ValueError("roles must not be empty.")
            known = {
                getattr(Constants.ChannelRoles, name)
                for name in dir(Constants.ChannelRoles)
                if not name.startswith("_")
            }
            unknown = sorted(set(roles) - known)
            if unknown:
                raise ValueError(
                    f"Unknown channel role(s) {unknown}. Known roles: "
                    f"{sorted(known)}"
                )
            kwargs.setdefault(self.Configuration.OptionalKeys.ROLES, roles)
        if montage is not None:
            kwargs.setdefault(
                self.Configuration.OptionalKeys.SYSTEM, montage.system
            )

        super().__init__(**kwargs)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Attach the description to the output context.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts carrying the description.

        Raises:
            ValueError: If a supplied list does not match the channel
                count, which would describe channels that do not exist.
        """
        port_context_out = super().setup(data, port_context_in)
        opt = self.Configuration.OptionalKeys

        count = channels.channel_count(port_context_in[PORT_IN])
        labels = self.config.get(opt.LABELS)
        roles = self.config.get(opt.ROLES)

        for name, value in (("labels", labels), ("roles", roles)):
            if value is not None and len(value) != count:
                raise ValueError(
                    f"{name} has {len(value)} entries but the input has "
                    f"{count} channel(s)."
                )

        if roles is None:
            roles = channels.roles_of(port_context_in[PORT_IN])

        port_context_out[PORT_OUT].update(
            channels.describe(
                list(roles),
                list(labels) if labels else None,
                self.config.get(opt.SYSTEM),
            )
        )
        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Pass the frame through unchanged.

        Args:
            data: Input frame.

        Returns:
            The same frame on the output port.
        """
        return {PORT_OUT: data[PORT_IN]}
