from PySide6.QtWidgets import (QHeaderView, QLabel, QTableWidget,
                               QTableWidgetItem, QWidget)

from ...backend.pipeline import Pipeline
from .base.widget import Widget

# Update interval for performance monitoring in milliseconds
UPDATE_INTERVAL_MS = 1000


class PerformanceMonitor(Widget):
    """
    Visualization widget for real-time pipeline performance monitoring.

    Displays the current state, condition, and per-node load statistics
    of a g.Pype pipeline in a tabular format with automatic updates.
    """

    def __init__(self, pipeline: Pipeline):
        """
        Initialize the PerformanceMonitor widget.

        Args:
            pipeline (Pipeline): The pipeline object to monitor for
                performance metrics and state information.
        """
        # Initialize base widget with container and descriptive name
        super().__init__(widget=QWidget(), name="Performance Monitor")

        # Set custom update interval for performance monitoring
        self._timer.setInterval(UPDATE_INTERVAL_MS)

        # Store reference to the monitored pipeline
        self.pipeline = pipeline

        # Initialize UI components
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface components."""
        # Create status display labels
        self.state_label = QLabel("State: -")
        self.condition_label = QLabel("Condition: -")

        # Create load monitoring table with three columns
        self.load_table = QTableWidget(0, 3)
        self.load_table.setHorizontalHeaderLabels(
            ["Class", "Name", "Load (%)"]
        )

        # Configure table to stretch columns to fit available space
        self.load_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        # Add all components to the main layout
        self._layout.addWidget(self.state_label)
        self._layout.addWidget(self.condition_label)
        self._layout.addWidget(self.load_table)

    def _update(self):
        """
        Update the performance monitor display with current pipeline data.

        This method is called periodically by the widget timer to refresh
        the displayed state, condition, and load information from the
        monitored pipeline.
        """
        # Retrieve current pipeline metrics
        state = self.pipeline.get_state()
        condition = self.pipeline.get_condition()
        load_data = self.pipeline.get_load()

        # Update status labels with current values
        self.state_label.setText(f"State: {state}")
        self.condition_label.setText(f"Condition: {condition}")

        # Refresh the load table with current node performance data
        self._update_load_table(load_data)

    def _update_load_table(self, load_data):
        """
        Update the load table with performance data for each pipeline node.

        Args:
            load_data (list): List of dictionaries containing node performance
                information with keys 'class', 'name', and 'load'.
        """
        # Set table row count to match number of nodes
        self.load_table.setRowCount(len(load_data))

        # Populate table with node performance information
        for row, node in enumerate(load_data):
            # Create table items for each column
            class_item = QTableWidgetItem(node["class"])
            name_item = QTableWidgetItem(node["name"])
            load_item = QTableWidgetItem(f"{node['load']:.2f}")

            # Insert items into the table at appropriate positions
            self.load_table.setItem(row, 0, class_item)
            self.load_table.setItem(row, 1, name_item)
            self.load_table.setItem(row, 2, load_item)
