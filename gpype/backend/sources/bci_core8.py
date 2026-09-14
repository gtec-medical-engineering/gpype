"""The old name for :class:`~gpype.backend.sources.bci_core.BCICore`.

``BCICore8`` read as the name of the 8-channel product. It never was:
a BCI Core-4 speaks the same protocol, reports its channel map the same
way and is driven by exactly the same code, so the 8 named a device
while sitting on a node that already supported the whole family. The
node is ``BCICore`` now.

The old name keeps working, and this module is why it can. Three things
force a real subclass in a module of its own rather than a plain
assignment in ``bci_core``:

* A stored document records ``module`` and ``class`` separately, and is
  rebuilt with ``import_module(module)`` then ``getattr(module, class)``.
  A document written before the rename names ``bci_core8`` /
  ``BCICore8``, so that module has to keep existing and keep answering
  to that attribute.
* The lazy-import map in ``gpype/__init__.py`` carries one symbol per
  module, so two public names need two modules regardless.
* The serialization allow-list derives the permitted module set from
  that same map. A name reachable only as an alias inside another
  module would not be on it, and the document would be refused rather
  than merely renamed.

So the alias is a subclass, and it serializes as itself: a document
saved from a ``BCICore8`` still says ``BCICore8`` and still loads.
Nothing is silently rewritten under the user.
"""

from __future__ import annotations

import warnings

from .bci_core import BCICore

__all__ = ["BCICore8"]


class BCICore8(BCICore):
    """Deprecated alias for :class:`BCICore`.

    Behaves exactly like ``BCICore``; only the name is different. Use
    ``gp.BCICore`` instead.
    """

    def __init__(self, *args, **kwargs):
        """Warn, then build an ordinary BCICore.

        The warning is raised here rather than at import, because the
        lazy-import map imports this module to reach the attribute --
        so an import-time warning would fire for anyone who merely
        touched ``gp.BCICore8`` in a ``dir()``, and would *not* fire for
        a document that named it, since the module may already be
        loaded. Construction is the event worth reporting.

        ``stacklevel=2`` so the warning points at the line that built
        the node rather than at this file, which is the only line the
        reader can act on.
        """
        warnings.warn(
            "BCICore8 is deprecated and will be removed in a future "
            "release; use BCICore instead. The node drives the whole "
            "BCI Core family -- a Core-4 as much as a Core-8 -- so the "
            "8 named a device rather than the node. Behaviour is "
            "unchanged.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
