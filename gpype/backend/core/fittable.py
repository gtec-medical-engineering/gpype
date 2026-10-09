"""Fittable: fit offline on a whole recording, deploy the result online.

A fittable node learns something from a recording -- a per-channel mean,
a spatial filter, a classifier's weights -- once, in a batch run, and then
applies exactly that, unchanged, in a realtime deployment. The learning
half and the applying half are two different methods precisely so that
neither can silently drift from the other:

``fit(block, context) -> state``
    Runs once, on the whole recording, during :meth:`Pipeline.fit`. Never
    called on a realtime path.
``load_state(state)``
    Adopts a state this node did not fit itself -- one loaded from a
    document, or the one :meth:`fit` just returned. Rebuilds whatever
    internal, non-serialisable form the node computes from.
``apply_block(block)``
    Produces this node's output for a whole block, from the currently
    loaded state. Called by :meth:`process` after fitting (or instead of
    it, for a batch run that only applies an already-loaded artifact).

A concrete node still writes its own ``step()`` for the realtime,
frame-at-a-time path; nothing here can write it generically, because a
transform's frame shape and its batch shape are not always the same
(:class:`~gpype.Sklearn`'s classifier mode is the example). What *is*
generic, and lives here, is fitting itself, the state contract, and the
provenance a fitted artifact carries forward.

State is a flat ``dict``: entries are either numeric ``numpy`` arrays or
plain JSON-safe values (D-BATCH-91) -- never pickle, never an opaque
object. :mod:`gpype.common._private.artifacts` is what turns that dict
into the document's ``artifacts`` section and back; a node author never
sees that encoding.

Class hierarchy: a concrete node inherits ``Fittable`` ahead of its node
base -- ``class Standardize(Fittable, IONode)`` -- so ``Fittable``'s
``process()`` is found before ``Node``'s raising default (D-BATCH-47:
``process_handler`` binds to whichever ``process`` the MRO resolves to).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import numpy as np

from ...common._private.naming import node_label
from ...common.constants import Constants

#: Node config keys excluded from :func:`_config_digest`: an artifact's
#: own reference to itself, its document id and port ids (never
#: hyperparameters, and never stable across two documents that fit the
#: identical configuration), and ``edge_id`` (residency, not fitting).
_CONFIG_DIGEST_EXCLUDED = frozenset(
    {"artifact", "id", "name", "input_ports", "output_ports", "edge_id"}
)

#: Configuration key a fitted node's document entry carries: the
#: document's own top-level ``artifacts`` section is keyed by the same
#: string (see ``Pipeline.ARTIFACTS_KEY``). Declared as its own
#: standalone ``Configuration`` here -- not inheriting a node base's --
#: because ``common.config_keys.recognised_keys`` walks every class in
#: the MRO independently and unions what each declares; a concrete node
#: such as ``Standardize`` need not repeat it.
ARTIFACT_KEY = "artifact"

#: Trust values that carry an artifact's mark forward (D-BATCH-40 travels
#: this in the port context; case-insensitive, since readers write it in
#: lower case).
_MARKING_TRUST_STATES = frozenset({"unsealed", "recovered"})


class Fittable:
    """Mixin: ``fit``/``load_state``/``apply_block``, plus ``process()``.

    Args:
        artifact: The document's reference to a previously fitted state
            for this node -- its ``sha256`` in the document's own
            ``artifacts`` section. ``None`` for a node that has not been
            fitted yet. Never resolved here: ``Pipeline.deserialize``
            reads the whole document, this node's config included, and
            calls :meth:`load_state` itself once the artifact is found
            and its digest verified.
        **kwargs: Forwarded to the node base this is mixed into.

    Note:
        Deliberately no nested ``Configuration`` here. ``self.Configuration``
        is looked up through the concrete node's MRO and *instantiated*
        (``Portable.create_config``), not merely read -- a standalone
        class here would come before the node base's real one and would
        be built instead of it. ``artifact`` still reaches
        ``common.config_keys.recognised_keys`` because that function
        also scans every ``__init__`` in the MRO for named parameters,
        and this one declares it.
    """

    def __init__(self, *args: Any, artifact: Optional[str] = None, **kwargs):
        super().__init__(*args, artifact=artifact, **kwargs)
        self._gpype_artifact_ref = artifact
        self._gpype_state: Optional[dict] = None
        self._gpype_marked: bool = False
        self._gpype_fit_run: bool = False
        self._gpype_run_verdict = None
        self._gpype_external_mark: Optional[str] = None
        self._gpype_input_context: dict = {}

    # -- the contract a concrete node implements -------------------------

    def fit(self, block: np.ndarray, context: dict) -> dict:
        """Learn this node's state from a whole recording.

        Called once, during :meth:`Pipeline.fit`, with the read-only
        block a batch run hands every node (D-BATCH-44) -- mutating it
        in place raises at the offending line rather than corrupting a
        sibling branch.

        Args:
            block: The complete input, shape ``(time, channel)`` or
                ``(time, channel, trial)`` downstream of
                :class:`~gpype.Epochs`.
            context: The input port's context at the time of fitting --
                sampling rate, channel roles, trials, trust, and
                whatever else the upstream pipeline published.

        Returns:
            A flat dict of numeric arrays and JSON-safe plain values.
            Framework keys (``marked``, and the channel-count and rate
            this mixin records) are added by :meth:`process`; a
            subclass returns only its own.

        Raises:
            NotImplementedError: Always, on this mixin. A concrete node
                overrides it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fit()."
        )

    def load_state(self, state: dict) -> None:
        """Adopt a state this node did not necessarily fit itself.

        Called with the same dict :meth:`fit` returned (fresh, in
        :meth:`process`) or one decoded from a document
        (``Pipeline.deserialize``). Rebuilds whatever internal form
        :meth:`apply_block` and ``step`` actually compute with -- a
        subclass typically keeps its arrays as instance attributes
        rather than re-reading the dict on every frame.

        Args:
            state: The flat state dict.

        Raises:
            NotImplementedError: Always, on this mixin. A concrete node
                overrides it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement load_state()."
        )

    def apply_block(self, block: np.ndarray) -> np.ndarray:
        """Produce this node's batch output from the currently loaded state.

        Args:
            block: The complete input.

        Returns:
            The transformed block.

        Raises:
            NotImplementedError: Always, on this mixin. A concrete node
                overrides it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement apply_block()."
        )

    # -- the state contract ----------------------------------------------

    @property
    def state(self) -> Optional[dict]:
        """The fitted state, or ``None`` before a fit or a load."""
        return self._gpype_state

    # -- framework wiring: entitlement, context, dispatch -----------------

    def attach_entitlement(self, verdict) -> None:
        """Learn this run's own resolved gate.

        Called by :meth:`Pipeline.start` on every node exposing this
        method (the same hook a sink uses to learn whether its output
        must carry the non-commercial mark), before any node's ``setup``
        runs. A fitted node reuses it for the same reason a sink does:
        fitting on an unlicensed or unattested machine marks the
        artifact this run produces (D-BATCH-92).

        Args:
            verdict: The run's resolved
                :class:`~gpype.common._private.entitlement.Entitlement`.
        """
        self._gpype_run_verdict = verdict

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Capture the input context, and refuse a state that disagrees.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts, from the rest of the MRO.

        Raises:
            ValueError: If a loaded state's recorded channel count or
                sampling rate disagrees with this input (D-BATCH-33):
                refused rather than silently broadcast.
        """
        port_context_out = super().setup(data, port_context_in)
        context = dict(port_context_in[Constants.Defaults.PORT_IN])
        self._gpype_input_context = context
        self._refuse_mismatched_state(context)
        return port_context_out

    def _refuse_mismatched_state(self, context: dict) -> None:
        """Part of :meth:`setup`; see its docstring."""
        state = self._gpype_state
        if state is None:
            return
        label = node_label(self)

        want_channels = state.get("_channel_count")
        have_channels = context.get(Constants.Keys.CHANNEL_COUNT)
        if (
            want_channels is not None
            and have_channels is not None
            and int(have_channels) != int(want_channels)
        ):
            raise ValueError(
                f"{label}: this state was fitted on {want_channels} "
                f"channel(s), but the input here has {have_channels}. "
                f"Refit on data with this shape rather than have "
                f"{label} silently broadcast a mismatch."
            )

        want_rate = state.get("_sampling_rate")
        have_rate = context.get(Constants.Keys.SAMPLING_RATE)
        if (
            want_rate is not None
            and have_rate is not None
            and float(have_rate) != float(want_rate)
        ):
            raise ValueError(
                f"{label}: this state was fitted at {float(want_rate):g} "
                f"Hz, but the input here is {float(have_rate):g} Hz."
            )

        want_digest = state.get("_config_digest")
        if want_digest is not None and want_digest != self._config_digest():
            raise ValueError(
                f"{label}: this artifact was fitted with a different "
                f"configuration than {label} declares now -- refit "
                f"rather than deploy a stale fit. Loading or starting "
                f"refuses it instead of silently applying a state that "
                f"no longer matches what this node would learn "
                f"(D-BATCH-103)."
            )

    def _config_digest(self) -> str:
        """Hash of this node's own configuration, for refit detection.

        Excludes the artifact reference itself, document/port ids and
        ``edge_id`` (:data:`_CONFIG_DIGEST_EXCLUDED`) -- none of those
        are a hyperparameter a refit would need to reflect, and the
        first two are never stable across two documents that fit the
        identical configuration in the first place. Recorded into the
        state at fit time (:meth:`_fit_and_adopt`) and compared again on
        every :meth:`setup` (D-BATCH-103, LQ-F5): editing a node's
        configuration after fitting changes this digest, and a loaded
        state whose recorded digest disagrees is refused rather than
        silently applied.

        Returns:
            Hex SHA-256 of the surviving configuration's canonical JSON.
        """
        config = {
            key: value
            for key, value in dict(self.config).items()
            if key not in _CONFIG_DIGEST_EXCLUDED
        }
        canonical = json.dumps(
            config, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def process(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Fit (when this run is a fit run), then apply the state.

        Args:
            data: The complete input, under
                ``Constants.Defaults.PORT_IN``.

        Returns:
            :meth:`apply_block`'s output, under
            ``Constants.Defaults.PORT_OUT``.

        Raises:
            ValueError: If this is not a fit run and no state was ever
                fitted or loaded.
        """
        port_in = Constants.Defaults.PORT_IN
        port_out = Constants.Defaults.PORT_OUT
        block = data[port_in]

        if self._gpype_fit_run:
            self._fit_and_adopt(block)
        elif self._gpype_state is None:
            label = node_label(self)
            raise ValueError(
                f"{label} has no fitted state. Call Pipeline.fit() "
                f"first, or load a document whose artifacts section "
                f"already carries one for {label}."
            )

        return {port_out: self.apply_block(block)}

    def fit_offline(
        self, block: np.ndarray, context: Optional[dict] = None
    ) -> dict:
        """Fit directly on *block*, without a running ``Pipeline``.

        :meth:`Pipeline.fit` is the sanctioned route and covers the
        ordinary case -- one batch source feeding this node. This is the
        escape hatch for training data that does not fit that shape,
        such as concatenating epochs cut from more than one recording (a
        batch pipeline takes exactly one source and merges no inputs,
        so two conditions cut with two :class:`~gpype.Epochs` cannot
        both feed one node in the same pipeline). It runs exactly what
        :meth:`process` runs during a fit run, with no pipeline, no
        cycle, and no entitlement gate around it -- the caller supplies
        whatever context :meth:`fit` needs directly. With no gate, the
        artifact is marked (D-BATCH-106). One from
        :meth:`Pipeline.fit` is marked where its run is: without a
        licence, or from marked data (D-ENT-96, D-ENT-97).

        Args:
            block: The whole training block.
            context: Whatever this node's :meth:`fit` reads -- at least
                the sampling rate, and, for a node that reads it,
                ``Constants.Keys.TRIALS``. Defaults to empty.

        Returns:
            The adopted state, the same object now on :attr:`state`.
        """
        self._gpype_input_context = dict(context or {})
        self._fit_and_adopt(block)
        return self._gpype_state

    def _fit_and_adopt(self, block: np.ndarray) -> None:
        """Run the subclass's ``fit``, stamp provenance, then load it.

        Args:
            block: The complete input this run is fitting on.
        """
        context = dict(self._gpype_input_context)
        state = dict(self.fit(block, context))
        state["marked"] = self._fit_is_marked(context)
        state.setdefault(
            "_channel_count",
            int(block.shape[1]) if block.ndim >= 2 else None,
        )
        rate = context.get(Constants.Keys.SAMPLING_RATE)
        if rate is not None:
            state.setdefault("_sampling_rate", float(rate))
        # LQ-F5, D-BATCH-103: what this fit could later be checked
        # against -- this node's own configuration, so a config edited
        # after fitting is refused rather than silently applied, and
        # the training input's identity (LQ-P1), so the provenance
        # record of a later deployment can name what trained it.
        state["_config_digest"] = self._config_digest()
        training_input = context.get(Constants.Keys.INPUT)
        if training_input is not None:
            state["_training_input"] = dict(training_input)
        self._gpype_state = state
        self._gpype_marked = bool(state.get("marked", False))
        self.load_state(state)

    def _fit_is_marked(self, context: dict) -> bool:
        """Whether the artifact this fit produces must carry the mark.

        True when any of D-BATCH-92's three conditions holds: this
        fitting run's own resolved entitlement was already marked
        (:meth:`attach_entitlement`), or there is none; the input's trust
        (``Constants.Keys.TRUST``) is ``unsealed`` or ``recovered``; or
        the single batch source this run read from carries the
        entitlement mark of the run that produced *it* (a
        :class:`~gpype.backend.sources.base.recording_reader.RecordingReader`'s
        own ``mark``, which :meth:`Pipeline.fit` reads before running
        and passes down -- it travels outside the port context, since
        nothing else in g.Pype has needed it there).

        Args:
            context: This fit's input context.

        Returns:
            Whether the fitted artifact must carry the mark.
        """
        verdict = self._gpype_run_verdict
        # No verdict is a fit no gate was resolved for -- fit_offline().
        # Read as unmarked, it trained an unmarked artifact on an
        # unlicensed machine; it fails closed instead (D-BATCH-106).
        if verdict is None or verdict.marked:
            return True
        trust = str(context.get(Constants.Keys.TRUST) or "").lower()
        if trust in _MARKING_TRUST_STATES:
            return True
        return bool(self._gpype_external_mark)
