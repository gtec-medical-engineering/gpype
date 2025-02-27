from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QGroupBox, QVBoxLayout, QBoxLayout
)
from typing import Type
from abc import abstractmethod
from PySide6.QtCore import QTimer

UPDATE_INTERVAL_MS: int = 50


class Widget:
    def __init__(self, widget: QWidget, name: str = "",
                 layout: Type[QBoxLayout] = QVBoxLayout):
        self.widget = widget
        self._timer = QTimer()
        self._timer.timeout.connect(self._update)

        box_layout = QHBoxLayout()
        box = QGroupBox(name)
        box_layout.addWidget(box)
        self._layout: Type[QBoxLayout] = layout(box)
        box.setLayout(self._layout)
        self.widget.setLayout(box_layout)

    def run(self):
        self._timer.start(UPDATE_INTERVAL_MS)

    def terminate(self):
        self._timer.stop()

    @abstractmethod
    def _update(self):
        pass
