from PySide6.QtWidgets import (
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QWidget
)
from .base.widget import Widget
from ...backend.pipeline import Pipeline

UPDATE_INTERVAL_MS = 1000


class PerformanceMonitor(Widget):
    def __init__(self, pipeline: Pipeline):
        """
        Widget-based Performance Monitor for monitoring pipeline state,
        condition, and load.

        Args:
            pipeline: Pipeline object to monitor.
        """
        Widget.__init__(self, widget=QWidget(), name="Performance Monitor")

        self.pipeline = pipeline

        # Create UI elements
        self.state_label = QLabel("State: -")
        self.condition_label = QLabel("Condition: -")

        self.load_table = QTableWidget(0, 3)
        self.load_table.setHorizontalHeaderLabels(["Class", "Name", "Load (%)"])  # noqa: E501
        self.load_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)  # noqa: E501

        # Layout setup

        self._layout.addWidget(self.state_label)
        self._layout.addWidget(self.condition_label)
        self._layout.addWidget(self.load_table)

    def _update(self):
        """
        Updates the widget with the current pipeline load.
        """
        # Fetch data from the pipeline
        state = self.pipeline.get_state()
        condition = self.pipeline.get_condition()
        load = self.pipeline.get_load()

        # Update labels
        self.state_label.setText(f"State: {state}")
        self.condition_label.setText(f"Condition: {condition}")

        # Update load table
        self.load_table.setRowCount(len(load))
        for row, node in enumerate(load):
            class_item = QTableWidgetItem(node["class"])
            name_item = QTableWidgetItem(node["name"])
            load_item = QTableWidgetItem(f"{node['load']:.2f}")

            self.load_table.setItem(row, 0, class_item)
            self.load_table.setItem(row, 1, name_item)
            self.load_table.setItem(row, 2, load_item)
