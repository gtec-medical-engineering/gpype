"""Application-wide theming for g.Pype frontends.

The dark look is two things, not one. The style sheet in
``resources/style`` carries the geometry and the colours it states
literally, but 28 of its declarations read ``palette(...)`` -- so
without a matching dark ``QPalette`` it resolves those against the
system palette, and the result is a light window with dark trim.

The palette is also the only way to reach the plots. A ``Scope`` takes
its curve colour from ``ColorRole.WindowText`` and its plot background
from ``ColorRole.Window`` *when it is constructed*, because a style
sheet cannot touch a pyqtgraph plot -- that is a QGraphicsView drawing
QGraphicsItems, not styled widgets. So the palette has to be on the
QApplication before the first widget exists, which is why this is
applied from ``MainApp.__init__`` and not left to the caller.

The sheet's icons are written as Qt resource paths --
``url(:/Resources/style/icon_close.png)`` -- which resolve only
out of a compiled ``.qrc``. Rather than add a resource-compilation
step to the build, the loader rewrites those URLs to the icon files as
they ship beside the sheet.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Theme applying the bundled dark palette and style sheet.
DARK = "dark"

#: Theme leaving Qt's own defaults untouched.
SYSTEM = "system"

#: Every accepted theme name.
THEMES = (DARK, SYSTEM)

#: Directory holding style.qss and the icons it references. It
#: reaches the wheel through [tool.setuptools.package-data] on
#: "gpype.frontend", which ships resources/**/* -- the same rule that
#: carries the window icon.
STYLE_DIR = Path(__file__).parent / "resources" / "style"

#: The sheet's icon references, as Qt resource paths. The file name is
#: captured so the rewrite can point it at STYLE_DIR.
_ICON_URL = re.compile(r"url\(\s*:/Resources/style/([^)\s]+?)\s*\)")

#: Corrections for widgets g.Pype itself puts on screen. Appended
#: rather than edited into the sheet, so the sheet stays what it
#: shipped as -- bar its resource prefix, renamed along with the
#: directory -- and every deviation from it states its reason here.
_OVERRIDES = """
/* The scope's -/+ steppers are a fixed 24x24, and the sheet asks for
   12px of padding on each side -- which leaves zero for the glyph, so
   both buttons render blank. The property is set where they are
   built, in the time series scope core. */
QPushButton[compact="true"]{
  padding-left:0px;
  padding-right:0px;
}
"""


def stylesheet(directory: Path = STYLE_DIR) -> str:
    """Read the dark style sheet with its icon URLs made resolvable.

    Args:
        directory: Directory holding ``style.qss`` and its icons.

    Returns:
        str: The style sheet, or an empty string when the sheet is not
            readable. An install whose resources went missing should
            look unstyled, not raise out of a constructor.
    """
    try:
        qss = (directory / "style.qss").read_text(encoding="utf-8")
    except OSError:
        return ""

    # as_posix(), and quoted: Qt reads a backslash in a style sheet as
    # an escape, and an unquoted url() ends at the first space -- which
    # any install under "C:/Program Files/..." would supply.
    base = directory.as_posix()
    qss = _ICON_URL.sub(lambda m: f'url("{base}/{m.group(1)}")', qss)
    return qss + _OVERRIDES


def dark_palette():
    """Build the dark palette the style sheet is written against.

    Only the roles the sheet and the widgets actually read are set.
    The rest keep Qt's defaults on purpose: ``Mid``, for one, is what
    the sheet uses for status-bar text, and a light grey is what makes
    that legible on the dark gradient behind it.

    Returns:
        QPalette: The palette to install on the QApplication.
    """
    from PySide6.QtGui import QColor, QPalette

    role = QPalette.ColorRole
    disabled = QPalette.ColorGroup.Disabled

    white = QColor(255, 255, 255)
    grey = QColor(127, 127, 127)

    palette = QPalette()
    palette.setColor(role.Window, QColor(53, 53, 53))
    palette.setColor(role.WindowText, white)
    palette.setColor(disabled, role.WindowText, grey)
    palette.setColor(role.Base, QColor(42, 42, 42))
    palette.setColor(role.AlternateBase, QColor(66, 66, 66))
    palette.setColor(role.ToolTipBase, white)
    palette.setColor(role.ToolTipText, QColor(53, 53, 53))
    palette.setColor(role.Text, white)
    palette.setColor(disabled, role.Text, grey)
    palette.setColor(role.Dark, QColor(35, 35, 35))
    palette.setColor(role.Shadow, QColor(20, 20, 20))
    palette.setColor(role.Button, QColor(53, 53, 53))
    palette.setColor(role.ButtonText, white)
    palette.setColor(disabled, role.ButtonText, grey)
    palette.setColor(role.BrightText, QColor(255, 0, 0))
    palette.setColor(role.Link, QColor(42, 130, 218))
    palette.setColor(role.Highlight, QColor(42, 130, 218))
    palette.setColor(disabled, role.Highlight, QColor(80, 80, 80))
    palette.setColor(role.HighlightedText, white)
    palette.setColor(disabled, role.HighlightedText, grey)
    return palette


def apply(app, theme: str = DARK) -> None:
    """Apply a theme to a QApplication.

    Args:
        app: The QApplication to style.
        theme: One of THEMES. ``SYSTEM`` leaves Qt alone, which is
            what a host application bringing its own look wants.

    Raises:
        ValueError: If the theme is not one of THEMES.
    """
    if theme not in THEMES:
        raise ValueError(
            f"unknown theme {theme!r}; expected one of {list(THEMES)}"
        )

    if theme == SYSTEM:
        return

    # Fusion first: setStyle() installs the style's standard palette,
    # so setting ours before it would be undone. Fusion is also what
    # the sheet is drawn against -- the native Windows style ignores
    # much of what it asks for.
    #
    # Called through the instance rather than the QApplication class so
    # that a caller handing in a stand-in (the test suite passes a
    # mock) does not have the real process style changed underneath it.
    app.setStyle("Fusion")
    app.setPalette(dark_palette())

    qss = stylesheet()
    if qss:
        app.setStyleSheet(qss)
