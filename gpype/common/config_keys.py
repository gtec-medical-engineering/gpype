"""Which configuration keys a class actually recognises.

A configuration accepts arbitrary keys. A misspelt or outdated one is
stored, kept, and written back out on the next save while the node
quietly runs on its default -- easy to miss by hand, and close to
guaranteed with a machine-generated document, where nobody reads the
configuration at all. It was measured: a document with ``sampling_rat:
500`` loads, runs at the default rate, and round-trips with the
misspelling intact.

The check used to live inside ``Node``, as a method reading
``self._target``. That put it out of reach of the two callers that need
it most:

* a **chain**. ``Generator`` is an ``ioc.OChain``, not a g.Pype ``Node``,
  and a chain's configuration is what carries the parameters in a
  document -- so the one class whose keys a document actually sets was
  never checked. In server residency the core that would have been
  checked is not built at all.
* a **runtime** answering "is this document valid?" before running it.
  Reimplementing the rule there would give two rules that agree until
  one of them changes.

So the rule lives here, as a function of a class and a mapping, and
``Node`` calls it like anybody else.
"""

from __future__ import annotations

import inspect
from typing import Iterable, Mapping

#: Nested classes that declare recognised keys. ``EmptyableKeys`` is
#: included because it exists -- the scopes use it for keys that may be
#: absent but may also legitimately be empty -- and its entries were
#: recognised only by accident: every one of them happens to also be a
#: constructor parameter. The first that was not would have been
#: reported as unrecognised.
KEY_HOLDERS = ("Keys", "OptionalKeys", "EmptyableKeys")

#: Keys the framework itself sets, which no node declares.
#:
#: ``stream_id`` is derived by ``chain_params.stream_id_for`` and passed
#: through ``**kwargs`` into the chain's configuration, so it appears in
#: neither a Keys holder nor a constructor signature. It is declared as a
#: key only on the private ``Link`` node.
#:
#: Measured rather than guessed: the rule was run against every node in
#: the package, built cleanly with no stray keys, and this was the only
#: thing it reported -- on 14 of 36 classes. Anything reported for a
#: clean build is a false positive by definition, and a validator that
#: cries wolf on a correct document is worse than none.
FRAMEWORK_KEYS = frozenset({"stream_id"})


def recognised_keys(cls: type) -> set:
    """Every configuration key *cls* is known to accept.

    Recognised means declared in any of :data:`KEY_HOLDERS` along the
    class hierarchy, named as a constructor parameter anywhere in it, or
    listed in :data:`FRAMEWORK_KEYS`. Constructor parameters count
    because most nodes accept their parameters that way and never declare
    them as keys.

    Args:
        cls: The class whose configuration is in question.

    Returns:
        The recognised key names.
    """
    known: set = set(FRAMEWORK_KEYS)
    for base in getattr(cls, "__mro__", (cls,)):
        configuration = base.__dict__.get("Configuration")
        for holder in KEY_HOLDERS:
            declared = (
                getattr(configuration, holder, None) if configuration else None
            )
            if declared is None:
                continue
            known.update(
                getattr(declared, name)
                for name in dir(declared)
                if not name.startswith("_")
            )
        init = base.__dict__.get("__init__")
        if init is not None:
            try:
                known.update(inspect.signature(init).parameters)
            except (TypeError, ValueError):  # pragma: no cover
                pass
    return known


def unrecognised_keys(cls: type, config: Mapping) -> list:
    """The keys in *config* that *cls* does not recognise.

    Reported, never rejected: a class is free to accept something this
    cannot see, and refusing would break it. Keys beginning with an
    underscore are ignored as deliberately private.

    Args:
        cls: The class the configuration belongs to.
        config: The configuration.

    Returns:
        The unrecognised keys, sorted, so a message about them is stable.
    """
    if not isinstance(config, Mapping):
        return []
    known = recognised_keys(cls)
    return sorted(
        key
        for key in config
        if key not in known and not str(key).startswith("_")
    )


def describe(cls: type, unknown: Iterable) -> str:
    """The sentence to show when keys are unrecognised.

    One wording, so the runtime's answer to "is this document valid" and
    the node's own log say the same thing about the same document.

    Args:
        cls: The class the configuration belongs to.
        unknown: The unrecognised keys.

    Returns:
        A message naming the class and the keys.
    """
    from ._private.naming import public_name

    return (
        f"Configuration of '{public_name(cls.__name__)}' contains "
        f"unrecognised key(s) {sorted(unknown)}; they are stored and "
        f"travel with the configuration, but nothing reads them."
    )
