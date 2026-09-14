from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ...common._private import channels
from ...common.constants import Constants
from ...common.montage import Montage
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class ChannelSelector(IONode):
    """Selects channels by name, by role, or by position.

    Router addresses channels by position, which is what you want when
    wiring a pipeline and not what you want when a montage changes. This
    addresses them by what they are:

        ChannelSelector(labels=["C3", "Cz", "C4"])
        ChannelSelector(roles=Constants.ChannelRoles.SIGNAL)
        ChannelSelector(indices=[0, 2, 4], exclude=True)

    Selecting by role is the device-agnostic form: every amplifier
    appends different extras, and asking for the signal channels is the
    same request on all of them.

    Every failure is raised at setup, i.e. when the pipeline starts and
    before a file or a socket exists. Skipping a channel that could not
    be found would change the channel count silently, everything
    downstream would accept it, and the operator would find out on
    analysis day that an electrode is missing from the whole recording.
    """

    #: Exactly one of these must be given -- the constructor refuses
    #: any other count. Declared because no per-parameter field can say
    #: it: read one at a time they all look merely optional, so a form
    #: built from the catalog would offer three optional fields and
    #: produce a document that always fails.
    ONE_OF = (("labels", "roles", "indices"),)

    class Configuration(IONode.Configuration):
        """Configuration class for ChannelSelector parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for ChannelSelector settings."""

            #: Whether the selection is inverted
            EXCLUDE = "exclude"

        class OptionalKeys(IONode.Configuration.OptionalKeys):
            """Optional configuration keys."""

            #: Channel labels to select
            LABELS = "labels"
            #: Channel roles to select
            ROLES = "roles"
            #: Channel indices to select
            INDICES = "indices"

    def __init__(
        self,
        labels: Optional[list] = None,
        roles: Optional[Union[str, list]] = None,
        indices: Optional[list] = None,
        exclude: bool = False,
        **kwargs,
    ):
        """Initialize the channel selector.

        Args:
            labels: Channel names to select. Also reorders: the output
                follows the order given.
            roles: One role, or several, from Constants.ChannelRoles.
            indices: Channel positions to select. Also reorders.
            exclude: Invert the selection. Survivors keep input order.
            **kwargs: Additional arguments for the parent IONode.

        Raises:
            ValueError: If not exactly one criterion is given, if it is
                empty, or if it repeats an entry.
        """
        given = [c for c in (labels, roles, indices) if c is not None]
        if len(given) != 1:
            raise ValueError(
                "Give exactly one of 'labels', 'roles' or 'indices'."
            )

        if roles is not None and isinstance(roles, str):
            roles = [roles]

        criterion = given[0] if not isinstance(given[0], str) else roles
        if len(criterion) == 0:
            raise ValueError("The selection must not be empty.")
        if len(set(map(str, criterion))) != len(criterion):
            raise ValueError("The selection must not repeat an entry.")

        opt = self.Configuration.OptionalKeys
        if labels is not None:
            # Normalise through the montage so an older name selects the
            # channel it became, e.g. T3 selecting T7.
            kwargs.setdefault(
                opt.LABELS, [Montage.normalise(str(x)) for x in labels]
            )
        if roles is not None:
            kwargs.setdefault(opt.ROLES, list(roles))
        if indices is not None:
            kwargs.setdefault(opt.INDICES, [int(x) for x in indices])

        super().__init__(exclude=bool(exclude), **kwargs)
        self._selection = None  # Resolved channel positions

    def _resolve(self, context: dict) -> list:
        """Resolve the configured criterion to channel positions.

        Args:
            context: Input port context.

        Returns:
            Channel positions, in selection order.

        Raises:
            ValueError: If anything requested is absent.
        """
        opt = self.Configuration.OptionalKeys
        count = channels.channel_count(context)

        wanted_labels = self.config.get(opt.LABELS)
        if wanted_labels:
            if not channels.has_labels(context):
                raise ValueError(
                    "Cannot select by label: the input declares no channel "
                    "labels. Use indices, or name the channels first with "
                    "a ChannelLabeler."
                )
            present = [
                Montage.normalise(str(x)) for x in channels.labels_of(context)
            ]
            missing = [x for x in wanted_labels if x not in present]
            if missing:
                raise ValueError(
                    f"Channel label(s) {missing} are not in the input. "
                    f"Available: {present}"
                )
            return [present.index(x) for x in wanted_labels]

        wanted_roles = self.config.get(opt.ROLES)
        if wanted_roles:
            present = list(channels.roles_of(context))
            missing = [r for r in wanted_roles if r not in present]
            if missing:
                raise ValueError(
                    f"No channel carries role(s) {missing}. Present roles: "
                    f"{sorted(set(present))}"
                )
            return [i for i, r in enumerate(present) if r in wanted_roles]

        wanted = self.config.get(opt.INDICES)
        out_of_range = [i for i in wanted if not 0 <= i < count]
        if out_of_range:
            raise ValueError(
                f"Channel index/indices {out_of_range} are out of range for "
                f"an input with {count} channel(s)."
            )
        return list(wanted)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Resolve the selection and describe the output channels.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts.

        Raises:
            ValueError: If the selection resolves to no channels at all.
        """
        port_context_out = super().setup(data, port_context_in)
        context = port_context_in[PORT_IN]

        selected = self._resolve(context)
        if self.config[self.Configuration.Keys.EXCLUDE]:
            dropped = set(selected)
            selected = [
                i
                for i in range(channels.channel_count(context))
                if i not in dropped
            ]
        if not selected:
            raise ValueError(
                "The selection leaves no channels; a node cannot emit an "
                "empty frame."
            )
        self._selection = selected

        out = port_context_out[PORT_OUT]
        out[Constants.Keys.CHANNEL_COUNT] = len(selected)
        out.update(channels.select(context, selected))
        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Emit the selected channels.

        Args:
            data: Input frame.

        Returns:
            The selected columns, in selection order.
        """
        return {PORT_OUT: data[PORT_IN][:, self._selection]}
