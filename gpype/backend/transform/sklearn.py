"""gp.Sklearn: bridge one scikit-learn estimator into fit offline, deploy
online.

scikit-learn is imported lazily and is not one of gpype's own
dependencies or extras (D-BATCH-94). Constructing or deserializing this
node never imports it: every process reading a document constructs every
node, and only the one that sets this node up, or fits it, needs
scikit-learn. There a missing package raises an ``ImportError`` naming
it, the same way ``gpype.__init__``'s ``_missing_extra_hint`` names a
missing extra.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...common._private import channels
from ...common._private.naming import node_label
from ...common.constants import Constants
from ..core.fittable import Fittable
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT

#: The scikit-learn module, imported on first use. ``None`` at module
#: scope so importing this file -- and constructing nothing -- never
#: requires scikit-learn to be installed.
sklearn = None


def _load() -> None:
    """Import scikit-learn, once, on first use.

    Raises:
        ImportError: Naming the package and how to install it, if it is
            not there. scikit-learn stays optional on purpose (D-BATCH-94):
            it is not declared anywhere in ``pyproject.toml``, so a plain
            ``pip install gpype`` -- or any of its extras -- never pulls
            it in.
    """
    global sklearn
    if sklearn is None:
        try:
            import sklearn as _sklearn
        except ImportError as error:
            raise ImportError(
                "gp.Sklearn needs scikit-learn, which is not installed. "
                "It is not one of gpype's own dependencies or extras:\n"
                "    pip install scikit-learn\n"
            ) from error
        sklearn = _sklearn


def _allowed_classes() -> dict:
    """The fixed allow-list of estimator classes this bridge accepts.

    Returns:
        Class name to class, imported lazily.
    """
    _load()
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    return {
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "PCA": PCA,
    }


#: The allow-list, by name: checked at construction without importing
#: scikit-learn (D-BATCH-108).
_ALLOWED = (
    "LinearDiscriminantAnalysis",
    "LogisticRegression",
    "PCA",
    "StandardScaler",
)

#: Estimator classes this bridge fits as a classifier -- one flattened
#: row per trial, ``y`` from the trial labels, output the index of the
#: predicted class.
_CLASSIFIERS = frozenset({"LinearDiscriminantAnalysis", "LogisticRegression"})

#: Estimator classes this bridge fits as a transformer -- rows are
#: samples, not trials.
_TRANSFORMERS = frozenset({"StandardScaler", "PCA"})


def _is_plain_json(value) -> bool:
    """Whether *value* is a plain, JSON-safe value.

    Args:
        value: Anything -- typically one entry of
            ``estimator.get_params()``.

    Returns:
        True for ``str``, ``int``, ``float``, ``bool``, ``None``, and a
        ``list`` or ``dict`` built only from those, recursively. False
        for anything else -- a ``numpy`` scalar, a ``RandomState``, or
        any other object.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_plain_json(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_plain_json(item)
            for key, item in value.items()
        )
    return False


def _refuse_unless_plain_json_params(
    estimator_class: str, params: dict
) -> None:
    """Refuse the first hyperparameter that is not plain JSON.

    No object may enter a document (D-BATCH-91): a hyperparameter such
    as ``random_state`` can hold a ``numpy`` ``RandomState`` instance,
    which this catches before it ever reaches ``self.config``.

    Args:
        estimator_class: The estimator's bare class name, for the
            message.
        params: ``estimator.get_params()``, or a document's own
            ``params``.

    Raises:
        ValueError: Naming the offending parameter and its value, or
            when *params* is not a mapping at all.
    """
    if not isinstance(params, dict):
        raise ValueError(
            f"gp.Sklearn({estimator_class!r}): params must be a mapping "
            f"of hyperparameter names to plain JSON values, not "
            f"{type(params).__name__}."
        )
    for name, value in params.items():
        if not _is_plain_json(value):
            raise ValueError(
                f"gp.Sklearn({estimator_class!r}): hyperparameter "
                f"{name!r} is {value!r} ({type(value).__name__}), which "
                f"is not a plain JSON value (str, int, float, bool, "
                f"None, or a list/dict of those). No object may enter "
                f"a document."
            )


#: Fitted array attributes this bridge carries in the artifact, per
#: allow-listed class. Only those the installed scikit-learn actually
#: set are kept (a solver may skip one): checked with ``hasattr``, not
#: assumed.
_ARRAY_ATTRS = {
    "LinearDiscriminantAnalysis": (
        "coef_",
        "intercept_",
        "classes_",
        "means_",
        "priors_",
        "scalings_",
        "xbar_",
        "explained_variance_ratio_",
    ),
    "LogisticRegression": ("coef_", "intercept_", "classes_"),
    "StandardScaler": ("mean_", "scale_", "var_", "n_samples_seen_"),
    "PCA": (
        "components_",
        "mean_",
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
        "n_components_",
        "noise_variance_",
    ),
}


class Sklearn(Fittable, IONode):
    """Fits one allow-listed scikit-learn estimator, offline and online.

    The allow-list is fixed: ``LinearDiscriminantAnalysis`` and
    ``LogisticRegression`` (both from ``sklearn``) as classifiers,
    ``StandardScaler`` and ``PCA`` as transformers. Anything else is
    refused at construction, naming the four. Construction checks the
    name and imports nothing; the estimator is built where the node is
    set up or fitted.

    A classifier fits on the ``(time, channel, trial)`` block a batch
    run downstream of :class:`~gpype.Epochs` delivers, with ``y`` taken
    from ``Constants.Keys.TRIALS`` -- one flattened row per trial. Its
    output is one column: the index of the predicted class in
    ``estimator.classes_`` (kept in the fitted state for reference, as a
    plain list of strings when the labels are not numeric -- never as
    an array, since an artifact carries no text arrays). Deployed
    online, it takes one 2-D frame -- the trained epoch length, flattened
    the same way -- and emits one prediction per frame.

    A transformer fits and applies on plain 2-D rows (samples x
    channels), in both batch and realtime. Its output channels, and a
    classifier's, declare no unit: neither is in the input's.

    Args:
        estimator: An unfitted instance of one of the four allow-listed
            classes -- the ordinary, author-time form, whose own
            ``get_params()`` becomes part of the fitted state, and is
            also written into this node's own configuration under
            ``params`` -- so an unfitted node's hyperparameters still
            survive a document, not only a fitted one's. Or the class's
            bare name as a string, which is what a document supplies
            when this node is rebuilt: no live estimator object can
            cross that boundary, so a document-rebuilt node builds an
            instance of the named class from ``params`` (or a bare
            default where none was recorded), or the one
            :meth:`load_state` was given, when it is set up.
        **kwargs: Additional arguments for the parent :class:`IONode`.

    Raises:
        ValueError: If *estimator* names or is an instance of a class
            outside the allow-list, or if one of its hyperparameters is
            not a plain JSON value (str, int, float, bool, None, or a
            list/dict of those) -- no object may enter a document.
        ImportError: From :meth:`setup` or a fit, not construction, if
            scikit-learn is not installed.
    """

    class Configuration(IONode.Configuration):
        """Configuration class for Sklearn parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for Sklearn settings."""

            #: The allow-listed estimator class's bare name.
            ESTIMATOR = "estimator"

        class OptionalKeys(IONode.Configuration.OptionalKeys):
            """Optional configuration keys for Sklearn settings."""

            #: The unfitted estimator's own ``get_params()`` -- absent
            #: only where none was ever given (a document written
            #: before this key existed).
            PARAMS = "params"

    def __init__(self, estimator, **kwargs):
        params_key = self.Configuration.OptionalKeys.PARAMS
        if isinstance(estimator, str):
            estimator_class = estimator
            instance = None
        else:
            estimator_class = type(estimator).__name__
            # By name alone, a class of that name from anywhere passed.
            if not type(estimator).__module__.startswith("sklearn."):
                estimator_class = (
                    f"{type(estimator).__module__}.{estimator_class}"
                )
            instance = estimator
        if estimator_class not in _ALLOWED:
            raise ValueError(
                f"gp.Sklearn does not support {estimator_class!r}; "
                f"the allow-list is {sorted(_ALLOWED)}."
            )

        if instance is None:
            params = kwargs.pop(params_key, None)
        else:
            # An instance is the ordinary, author-time form: its own
            # hyperparameters, not whatever a stray `params` kwarg says.
            params = dict(instance.get_params())
            kwargs.pop(params_key, None)
        if params:
            _refuse_unless_plain_json_params(estimator_class, params)
            kwargs[params_key] = dict(params)

        super().__init__(estimator=estimator_class, **kwargs)
        self._estimator_class = estimator_class
        self._is_classifier = estimator_class in _CLASSIFIERS
        #: None until built: see :meth:`_built`.
        self._estimator = instance
        self._loaded_state: Optional[dict] = None

    def _built(self):
        """The estimator, built from the loaded state on first use.

        Not at construction or in :meth:`load_state`: both run in every
        process reading a document, and only this node's own process
        needs scikit-learn.

        The hyperparameters come from the loaded state's own ``params``
        where one was loaded (a fit, or an artifact read from a
        document) -- that takes precedence, as it did before. Where
        none was loaded yet, they come from this node's own
        configuration instead (:data:`Configuration.OptionalKeys.PARAMS`),
        which is how an unfitted node's hyperparameters -- set at
        construction from a live estimator's own ``get_params()`` --
        survive a document that was serialized before any fit.

        Returns:
            The estimator instance.

        Raises:
            ImportError: If scikit-learn is not installed.
        """
        if self._estimator is None:
            allowed = _allowed_classes()
            state = self._loaded_state or {}
            params = state.get("params")
            if not params:
                params = dict(
                    self.config.get(self.Configuration.OptionalKeys.PARAMS, {})
                    or {}
                )
            estimator = allowed[self._estimator_class](**params)
            for attr in _ARRAY_ATTRS[self._estimator_class]:
                if attr in state:
                    setattr(estimator, attr, np.asarray(state[attr]))
            self._estimator = estimator
        return self._estimator

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Declare this node's output shape.

        A classifier's output is one column -- the predicted class's
        index -- regardless of how many channels it was trained on.
        A transformer's output keeps its input's width, except PCA,
        which narrows it to the number of components it was fitted
        with.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts.
        """
        estimator = self._built()
        port_context_out = super().setup(data, port_context_in)
        out = port_context_out[PORT_OUT]

        if self._is_classifier:
            out[Constants.Keys.CHANNEL_COUNT] = 1
            out.update(
                channels.describe(
                    [Constants.ChannelRoles.SIGNAL], ["prediction"], None
                )
            )
        elif self._estimator_class == "PCA":
            width = getattr(estimator, "n_components_", None)
            if width:
                out[Constants.Keys.CHANNEL_COUNT] = int(width)
                out.update(
                    channels.describe(
                        [Constants.ChannelRoles.SIGNAL] * int(width),
                        None,
                        None,
                    )
                )
        # A class index, a z-score or a component is in none of the
        # input's units, and StandardScaler rescales every column.
        if out.get(Constants.Keys.CHANNEL_UNITS) is not None:
            count = int(out[Constants.Keys.CHANNEL_COUNT])
            out[Constants.Keys.CHANNEL_UNITS] = [None] * count
        return port_context_out

    def fit(self, block: np.ndarray, context: dict) -> dict:
        """Fit the estimator, as a classifier or as a transformer.

        Args:
            block: ``(time, channel, trial)`` downstream of
                :class:`~gpype.Epochs`, for a classifier; ``(samples,
                channels)`` for a transformer.
            context: For a classifier, must carry
                ``Constants.Keys.TRIALS``.

        Returns:
            The estimator's ``get_params()`` under ``"params"``, plus
            its fitted array attributes.

        Raises:
            ValueError: If a classifier's input is not 3-D, or the
                context carries no trials; if a transformer's input is
                not 2-D.
        """
        label = node_label(self)
        estimator = self._built()
        if self._is_classifier:
            if block.ndim != 3:
                raise ValueError(
                    f"{label}: a classifier fits on a (time, channel, "
                    f"trial) block -- connect it downstream of "
                    f"gp.Epochs. Got shape {block.shape}."
                )
            trials = context.get(Constants.Keys.TRIALS)
            if not trials:
                raise ValueError(
                    f"{label}: no trials in the input context. A "
                    f"classifier's labels come from "
                    f"Constants.Keys.TRIALS, which gp.Epochs publishes."
                )
            y = [str(entry[1]) for entry in trials]
            n_trials = block.shape[2]
            x = block.transpose(2, 0, 1).reshape(n_trials, -1)
            estimator.fit(x.astype(np.float64, copy=False), y)
        else:
            if block.ndim != 2:
                raise ValueError(
                    f"{label}: a transformer fits on (samples, "
                    f"channels) rows. Got shape {block.shape}."
                )
            estimator.fit(block.astype(np.float64, copy=False))
        return self._estimator_state()

    def _estimator_state(self) -> dict:
        """The fitted estimator's params and array attributes.

        Returns:
            ``{"params": {...}, <attr>: <array or plain list>, ...}``.
        """
        estimator = self._built()
        state: dict = {"params": dict(estimator.get_params())}
        for attr in _ARRAY_ATTRS[self._estimator_class]:
            if not hasattr(estimator, attr):
                continue
            value = np.asarray(getattr(estimator, attr))
            if attr == "classes_" and value.dtype.kind not in "biufc":
                # A text array is refused by the artifact codec
                # outright (D-BATCH-91); keep string labels as a plain
                # list in meta instead, purely for a reader's benefit --
                # prediction only ever emits the class's index.
                state[attr] = [str(v) for v in value.tolist()]
            else:
                state[attr] = value
        return state

    def load_state(self, state: dict) -> None:
        """Adopt a state; the estimator is rebuilt from it on first use.

        Deferred because a document's artifacts are loaded wherever the
        document is read (see :meth:`_built`).

        Args:
            state: As :meth:`_estimator_state` returned it, fresh or
                decoded from a document.
        """
        self._loaded_state = dict(state)
        self._estimator = None

    def _predict_indices(self, x: np.ndarray) -> np.ndarray:
        """The predicted class's position in ``classes_``, per row."""
        estimator = self._built()
        labels = estimator.predict(x)
        classes = estimator.classes_.tolist()
        return np.asarray(
            [classes.index(label) for label in labels.tolist()],
            dtype=np.int64,
        )

    def apply_block(self, block: np.ndarray) -> np.ndarray:
        """Predict or transform the whole block, for a batch run.

        Args:
            block: The complete input, the same shape :meth:`fit` took.

        Returns:
            One predicted class index per trial (a classifier), or the
            transformed rows.
        """
        if self._is_classifier:
            n_trials = block.shape[2]
            x = block.transpose(2, 0, 1).reshape(n_trials, -1)
            indices = self._predict_indices(x.astype(np.float64, copy=False))
            return indices.reshape(-1, 1).astype(Constants.DATA_TYPE)
        transformed = self._built().transform(
            block.astype(np.float64, copy=False)
        )
        return np.asarray(transformed, dtype=Constants.DATA_TYPE)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Predict or transform one realtime frame.

        A classifier's frame is one epoch -- the trained epoch length,
        flattened the same way training flattened each trial -- and
        this emits its single prediction.

        Args:
            data: Input frame.

        Returns:
            One row: the predicted class's index (a classifier), or
            the transformed frame.
        """
        frame = data[PORT_IN]
        if self._is_classifier:
            x = frame.reshape(1, -1).astype(np.float64, copy=False)
            index = self._predict_indices(x)
            return {PORT_OUT: index.reshape(1, 1).astype(Constants.DATA_TYPE)}
        transformed = self._built().transform(
            frame.astype(np.float64, copy=False)
        )
        return {PORT_OUT: np.asarray(transformed, dtype=Constants.DATA_TYPE)}
