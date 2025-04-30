from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout
)
from PySide6.QtGui import QIcon
from .widgets.base.widget import Widget
from pathlib import Path
import os


class MainApp(QApplication):

    DEFAULT_POSITION = [100, 100, 800, 600]
    ICON_PATH = os.path.join(Path(__file__).parent, "resources", "gtec.ico")

    def __init__(self,
                 caption: str = "g.Pype Application",
                 position: list[int] = None):
        super().__init__([])
        self._widgets: list[Widget] = []

        # Create and configure main window
        self._window = QMainWindow()
        self._window.setWindowTitle(caption)
        self.setWindowIcon(QIcon(MainApp.ICON_PATH))
        self._window.setWindowIcon(QIcon(MainApp.ICON_PATH))
        if position is None:
            position = MainApp.DEFAULT_POSITION
        self._window.setGeometry(*position)

        # Create a central widget and set it to the main window
        central_widget = QWidget(self._window)
        self._window.setCentralWidget(central_widget)

        # Create and set the QVBoxLayout
        self._layout = QVBoxLayout(central_widget)
        central_widget.setLayout(self._layout)

        self.aboutToQuit.connect(self._on_quit)

    def add_widget(self, widget: Widget):
        self._widgets.append(widget)
        self._layout.addWidget(widget.widget)

    def _on_quit(self):
        [w.terminate() for w in self._widgets]

    def run(self) -> int:
        self._window.show()
        [w.run() for w in self._widgets]
        return self.exec()
