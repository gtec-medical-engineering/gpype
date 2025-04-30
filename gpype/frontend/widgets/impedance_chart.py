from PySide6.QtWidgets import (
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QWidget
)
from PySide6.QtGui import QPalette, QFont
from .base.widget import Widget
from ...backend.sources.g_nautilus import GNautilus
import numpy as np
import pyqtgraph as pg
import time


class ImpedanceChart(Widget):

    _IMPEDANCE_RANGE_LOWER_LIMIT = 5e3
    _IMPEDANCE_RANGE_UPPER_LIMIT = 200e3
    _IMPEDANCE_HIGH = 210e3
    _IMPEDANCE_LOW = 5e3
    _IMPEDANCE_INVALID = 220e3
    _UPDATE_RATE = 1e-2
    _LABEL_HIGH = "OPEN"
    _LABEL_LOW = "LOW"
    _LABEL_INVALID = "INVALID"
    _MAPPING = {
        -1: _IMPEDANCE_LOW,
        -2: _IMPEDANCE_LOW,
        -3: _IMPEDANCE_HIGH,
        -4: _IMPEDANCE_HIGH,
        -5: _IMPEDANCE_INVALID,
        -6: _IMPEDANCE_INVALID,
        -7: _IMPEDANCE_INVALID,
        -8: _IMPEDANCE_INVALID,
        -9: _IMPEDANCE_INVALID,
        -10: _IMPEDANCE_INVALID,
        -11: _IMPEDANCE_INVALID,
        -12: _IMPEDANCE_INVALID,
    }

    _plot_item: pg.PlotItem
    _last_plot_update: float
    _last_impedance_update: float
    _update_label: pg.TextItem

    def __init__(self, amplifier: GNautilus):
        widget = pg.PlotWidget()
        widget.setBackground(widget.palette().color(QPalette.Window))
        Widget.__init__(self, widget=QWidget(), name="Impedance Plot")

        self._plot_item = widget.getPlotItem()
        self._plot_item.showGrid(x=True, y=True, alpha=0.3)
        self._plot_item.getViewBox().setMouseEnabled(x=False, y=False)
        self._plot_item.setLabel('bottom', 'Electrode Index')
        self._plot_item.setLabel('left', 'Impedance', units='kΩ')
        ymax = self._IMPEDANCE_INVALID / 1e3
        self._plot_item.setYRange(0, ymax, padding=0.1)

        self._layout.addWidget(widget)
        self._bar_graph = None
        self._text_items = []

        self._last_plot_update = None
        self._last_impedance_update = None
        self._update_label = None

        amplifier.start_impedance_check()
        self._amplifier = amplifier

    def terminate(self):
        self._amplifier.stop_impedance_check()
        super().terminate()

    def _update(self):

        # shorten notation
        Z_HI = ImpedanceChart._IMPEDANCE_HIGH
        Z_LO = ImpedanceChart._IMPEDANCE_LOW
        Z_INV = ImpedanceChart._IMPEDANCE_INVALID
        Z_HI_LIM = ImpedanceChart._IMPEDANCE_RANGE_UPPER_LIMIT
        Z_LO_LIM = ImpedanceChart._IMPEDANCE_RANGE_LOWER_LIMIT

        t = time.time()
        if self._last_plot_update is not None:
            if t - self._last_plot_update < ImpedanceChart._UPDATE_RATE:
                return
        self._last_plot_update = t

        # Fetch data from amplifier
        z, fresh = self._amplifier.get_impedance()
        if fresh:
            self._last_impedance_update = t
        t_elapsed = t - self._last_impedance_update

        # process values here
        z[np.isnan(z)] = Z_INV
        z[(z > 0) & (z < Z_LO_LIM)] = Z_LO
        z[(z > Z_HI_LIM) & (z < Z_INV)] = Z_HI
        for k, v in ImpedanceChart._MAPPING.items():
            z[z == k] = v

        # generate bar colors
        z_norm = (z - Z_LO) / (Z_HI - Z_LO)
        colors = []
        for val in z_norm.flatten():  # Flatten to ensure we have a 1D array
            col = (0, 0, 0, 255)  # invalid impedance => black
            if val < 0:
                # low impedance => green
                col = (0, 255, 0, 255)
            elif val < 1:
                # interpolate from red (high) to green (low)
                r = int(255 * val)
                g = int(255 * (1 - val))
                col = (r, g, 0, 255)
            colors.append(col)

        # Create the bar graph
        if self._bar_graph is None:
            self._bar_graph = pg.BarGraphItem(x=np.arange(len(z)) + 1,
                                              height=z / 1e3,
                                              width=0.8,
                                              brushes=colors,
                                              pen=None)
            self._plot_item.addItem(self._bar_graph)
        else:
            self._bar_graph.setOpts(height=z / 1e3,
                                    brushes=colors)

        # Remove previous labels
        for text_item in self._text_items:
            self._plot_item.removeItem(text_item)
        self._text_items.clear()

        # Add new labels
        for i, zi in enumerate(z):
            label = ""
            if zi == Z_HI:
                label = ImpedanceChart._LABEL_HIGH
            elif zi == Z_LO:
                label = ImpedanceChart._LABEL_LOW
            elif np.isnan(zi) or zi == Z_INV:
                label = ImpedanceChart._LABEL_INVALID
            else:
                label = f"{zi / 1e3:.2f} kΩ"
            text_item = pg.TextItem(text=label,
                                    color=(32, 32, 32),
                                    anchor=(0.5, 0))
            text_item.setPos(i + 1, -3)
            self._plot_item.addItem(text_item)
            self._text_items.append(text_item)

        # Update the update label
        update_text = f"Last update {int(t_elapsed)} s ago"
        if self._update_label is None:
            self._update_label = pg.TextItem(text=update_text,
                                             color=(128, 128, 128),
                                             anchor=(1, 0))
            self._update_label.setPos(len(z) + 0.4, Z_INV / 1e3 + 20)
            self._plot_item.addItem(self._update_label)
            font = QFont()
            font.setPointSize(8)
            self._update_label.setFont(font)
        else:
            self._update_label.setText(update_text)
