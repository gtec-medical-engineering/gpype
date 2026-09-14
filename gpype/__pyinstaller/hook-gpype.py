# -*- coding: utf-8 -*-
"""PyInstaller hook for gpype.

Registered through the ``pyinstaller40`` entry point, so it applies
automatically to any application frozen with g.Pype in it, including the
tools under ``apps/``.

Its job is to declare only what PyInstaller's own analysis cannot see.
That analysis is better than it looks: it walks bytecode, so an import
inside a function body is found, which covers ``websockets``,
``gtec_attest``, ``gtec_licensing.api`` and ``gtec_oscar``. PySide6,
pyqtgraph and scipy ship their own hooks, and ioiocore is imported at
module scope all over the backend. None of those need repeating here --
listing them only makes the hook look complete while the two things that
really break stay missing.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = [
    # node.py is compiled by Cython into node.pyd -- see [tool.gpype]
    # cython_files in pyproject.toml. PyInstaller cannot read an
    # extension module's bytecode, so `import psutil` at the top of that
    # file is invisible in exactly the builds that matter. It is visible
    # from source, which is why this only ever fails in a frozen app.
    "psutil",
]

# The whole public API is resolved lazily. gpype/__init__.py keeps a
# _LAZY_IMPORTS map of {module path: symbol} and __getattr__ calls
# __import__ on an f-string built from it, so nothing static can tell
# that gp.Generator means gpype.backend.sources.generator. Without this,
# a frozen app imports gpype successfully and then raises AttributeError
# on the first symbol the user asks for.
#
# This pulls in the hardware sources too, and with them the optional
# driver packages they import. That costs size for users who have no
# amplifier; the alternative is an app where gp.Generator does not exist,
# so the trade is not close.
hiddenimports += collect_submodules("gpype")

# Runtime data files. PyInstaller collects none of these by itself.
#
#   gpype           the window icon, frontend/resources/gtec.ico
#   gtec_oscar      OSCAR_LIVE.profile and the randomisation table, which
#                   gtec_oscar loads from inside its own package at
#                   runtime; an OSCAR-enabled pipeline fails without them
datas = collect_data_files("gpype")

try:
    datas += collect_data_files("gtec_oscar")
except Exception:
    # A build environment without OSCAR is legitimate: only pipelines
    # that enable it need the package, and gpype degrades to a clear
    # ImportError at that point rather than at import time.
    pass
