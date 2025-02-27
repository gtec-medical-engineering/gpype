import pyqtgraph as pg
from abc import abstractmethod
from ....backend.core.i_node import INode
from ....backend.core.i_port import IPort
from .widget import Widget

from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPalette

DEFAULT_COLOR = (32, 32, 172)


class Scope(INode, Widget):

    _curves: list[pg.PlotDataItem]
    _plot_item: pg.PlotItem

    def __init__(self,
                 input_ports: list[IPort.Configuration] = None,
                 update_rule: 'INode.UpdateRule' = None,
                 **kwargs):
        INode.__init__(self,
                       input_ports=input_ports,
                       update_rule=update_rule,
                       **kwargs)
        widget = pg.PlotWidget()
        widget.setBackground(widget.palette().color(QPalette.Window))
        Widget.__init__(self, widget=QWidget(), name="Scope")
        self._plot_item = widget.getPlotItem()
        self._plot_item.showGrid(x=True, y=True, alpha=0.3)
        self._plot_item.getViewBox().setMouseEnabled(x=False, y=False)

        self._curves = None
        self._data = None

        self._layout.addWidget(widget)

    def set_labels(self, x_label: str, y_label: str):
        self._plot_item.setLabel('bottom', x_label)
        self._plot_item.setLabel('left', y_label)

    def add_curve(self, color=pg.mkColor(DEFAULT_COLOR)):
        if self._curves is None:
            self._curves = []
        curve = self._plot_item.plot(pen=pg.mkPen(color))
        self._curves.append(curve)
        return curve

    @abstractmethod
    def _update(self):
        pass
