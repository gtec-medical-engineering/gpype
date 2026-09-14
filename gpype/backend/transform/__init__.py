# Re-exported so `node` has a module path of its own in the public
# API's lazy-import map. That map is keyed by module path and holds one
# symbol per module, and `apply.py` exports two -- `Apply` and `node`
# -- so the second needs a different route to be registered at all.
#
# The same shape as `backend/core/__init__.py`, which re-exports
# `action` and `controllable` for the same reason. Matching it rather
# than inventing a second convention: an earlier version of this
# defined `node` by hand in the package initialiser, on the mistaken
# grounds that the catalogue generator would describe a function as a
# node it could not construct. It would not -- `generate_catalog.py`
# tests `inspect.isclass` and skips everything else, which is why
# `action` and `controllable` are registered and absent from the
# catalogue.
from .apply import node

__all__ = ["node"]
