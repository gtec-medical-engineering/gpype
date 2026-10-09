"""How an application's windows are known to the desktop: the taskbar icon.

Two things decide the icon a window gets in the Windows taskbar, and a
Python application gets both wrong by default:

**Which application the window belongs to.** A script runs as
``python.exe``, and the taskbar groups every window of a process by that
process's application ID, which for ``python.exe`` is Python's: the
button shows Python's icon whatever icon the window sets. A script
therefore names its process, with an explicit AppUserModelID, before its
first window. A frozen application is its own executable and must not:
an explicit ID would part its windows from the shortcut and the pinned
button that start it.

**The window's icon.** A Qt window has the icon the application sets, or
Qt's default, never the executable's own. It must be the application's:
the one it names, else the one a build carries, which gpype-compiler's
startup hook names in :data:`ICON_ENV`, else, in a frozen application on
Windows, its executable's icon, else g.Pype's.

:func:`apply` does both, and every g.Pype window should come through it:
``MainApp`` calls it, and so should a tool that builds its own
QApplication. On macOS and Linux only the icon applies.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional, Union

#: Names the application's icon file, for a host that starts the
#: application: gpype-compiler's startup hook sets it to the icon the
#: build carries.
ICON_ENV = "GPYPE_APP_ICON"

#: g.Pype's icon, for an application that names none.
DEFAULT_ICON = Path(__file__).parent / "resources" / "gpype.ico"

PathLike = Union[str, Path]


def app_id(name: str) -> str:
    """Return the AppUserModelID for an application's name.

    Args:
        name: The application's name, such as its window caption.

    Returns:
        ``gtec.gPype.<name>``, without the characters an ID may not
        hold, and at most 128 characters long.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "", name) or "Application"
    return f"gtec.gPype.{slug}"[:128]


def name_process(identifier: str) -> bool:
    """Give a script's process its own taskbar identity, on Windows.

    Must run before the process shows its first window. Does nothing in a
    frozen application, or where the process already has an explicit ID,
    such as one its host set.

    Args:
        identifier: The AppUserModelID, see :func:`app_id`.

    Returns:
        True if the ID was set now.
    """
    if sys.platform != "win32" or getattr(sys, "frozen", False):
        return False
    try:
        import ctypes

        shell32 = ctypes.windll.shell32
        current = ctypes.c_void_p()
        found = shell32.GetCurrentProcessExplicitAppUserModelID(
            ctypes.byref(current)
        )
        if found == 0:  # S_OK: named already
            ctypes.windll.ole32.CoTaskMemFree(current)
            return False
        result = shell32.SetCurrentProcessExplicitAppUserModelID(
            ctypes.c_wchar_p(identifier)
        )
        return result == 0
    except (AttributeError, OSError):
        return False


def icon_path(
    icon: Optional[PathLike] = None, default: PathLike = DEFAULT_ICON
) -> Optional[Path]:
    """Return the application's icon file.

    Args:
        icon: The icon the application names itself.
        default: The icon when neither it nor :data:`ICON_ENV` names one.

    Returns:
        The first of ``icon``, :data:`ICON_ENV` and ``default`` that is a
        file, or None.
    """
    for candidate in (icon, os.environ.get(ICON_ENV), default):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def window_icon(
    app, icon: Optional[PathLike] = None, default: PathLike = DEFAULT_ICON
):
    """Return the application's icon as a QIcon of the app's own binding.

    The icon named, else :data:`ICON_ENV`'s; in a frozen application on
    Windows, else its executable's, which its shortcut and pin show too;
    else ``default``. A frozen application on macOS that names none keeps
    its bundle's icon, so there it is None.

    Args:
        app: The QApplication; its Qt binding builds the icon.
        icon: The icon file the application names.
        default: The icon when nothing else gives one.

    Returns:
        A QIcon, or None.
    """
    gui, widgets, core = _binding(app)
    path = icon_path(icon, default=None)
    if path is not None:
        return gui.QIcon(str(path))
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            return None
        if sys.platform == "win32":
            own = widgets.QFileIconProvider().icon(
                core.QFileInfo(sys.executable)
            )
            if not own.isNull():
                return own
    if default and Path(default).is_file():
        return gui.QIcon(str(default))
    return None


def _binding(app):
    """The QtGui, QtWidgets and QtCore of the binding ``app`` belongs to:
    a PySide6 icon cannot be set on a PyQt6 application."""
    import importlib

    name = type(app).__module__.split(".")[0]
    if name not in ("PySide6", "PyQt6", "PySide2", "PyQt5"):
        name = "PySide6"
    return tuple(
        importlib.import_module(f"{name}.{part}")
        for part in ("QtGui", "QtWidgets", "QtCore")
    )


def apply(app, name: str, icon: Optional[PathLike] = None) -> None:
    """Give an application its taskbar identity and its icon.

    Call before the first window is shown.

    Args:
        app: The QApplication.
        name: The application's name, for its ID (:func:`app_id`).
        icon: Its icon file; see :func:`window_icon` for the rest.
    """
    name_process(app_id(name))
    found = window_icon(app, icon)
    if found is not None:
        app.setWindowIcon(found)
