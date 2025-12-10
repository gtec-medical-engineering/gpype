from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np

from ....common.constants import Constants
from ...core.o_port import OPort
from .source import Source

# Convenience constant for default output port name
PORT_OUT = Constants.Defaults.PORT_OUT


class EventSource(Source):
    """Event-driven source for asynchronous data generation.

    Generates events in response to external triggers.
    """

    def __init__(self, **kwargs):
        """Initialize event source with asynchronous output configuration.

        Args:
            **kwargs: Additional arguments for parent Source class including
                channel_count, frame_size, and other configuration parameters.
        """
        # Extract output_ports from kwargs with default async configuration
        op_key = self.Configuration.Keys.OUTPUT_PORTS
        output_ports: list[OPort.Configuration] = kwargs.pop(
            op_key, [OPort.Configuration(timing=Constants.Timing.ASYNC)]
        )

        # Initialize parent Source with configuration
        Source.__init__(self, output_ports=output_ports, **kwargs)

        # Initialize data storage for output ports
        op_key = self.Configuration.Keys.OUTPUT_PORTS
        cc_key = self.Configuration.Keys.CHANNEL_COUNT
        name_key = OPort.Configuration.Keys.NAME
        self._data = {}

        # Pre-allocate data arrays for non-inherited channel configurations
        for op, cc in zip(self.config[op_key], self.config[cc_key]):
            if cc != Constants.INHERITED:
                # Create zero-filled array with proper data type
                self._data[op[name_key]] = np.zeros(
                    (1, cc), dtype=Constants.DATA_TYPE
                )

        # Initialize delay thread components if delay is configured
        if self.source_delay > 0:
            self._delay_thread_queue: Optional[queue.Queue] = None
            self._delay_thread: Optional[threading.Thread] = None
            self._delay_thread_running: Optional[bool] = None

    def start(self):
        """Start event source."""

        # Call parent start method first
        Source.start(self)

        # Initialize delay processing thread if needed
        if self.source_delay > 0:
            self._delay_thread_queue = queue.Queue()
            self._delay_thread = threading.Thread(
                target=self._timer_loop, daemon=True
            )
            self._delay_thread_running = True
            self._delay_thread.start()

        # Trigger initial cycle to prepare for events
        self.cycle()

    def stop(self):
        """Stop event source."""

        # Signal delay thread to stop
        if hasattr(self, "_delay_thread_running"):
            self._delay_thread_running = False

        # Call parent stop method
        Source.stop(self)

    def trigger(self, value, port_name=PORT_OUT):
        """Trigger an event with the specified value.

        Args:
            port_name: Name of the output port to trigger.
            value: Event value to be transmitted. Converted to appropriate
                data type and formatted as single-sample array.
        """
        # Create data array with the event value
        data = {}
        if not isinstance(port_name, list):
            port_name = [PORT_OUT]
            value = [value]
        for i in range(len(port_name)):
            pn = port_name[i]
            pv = value[i]
            data[pn] = np.array([[pv]], dtype=Constants.DATA_TYPE)

        if self.source_delay > 0:
            # Queue event with timestamp for delayed processing
            timestamp = time.monotonic()
            self._delay_thread_queue.put((timestamp, data))
        else:
            # Process event immediately
            self._data = data
            self.cycle()  # Trigger node cycle
            self._data = None  # Clear data after processing

    def _timer_loop(self):
        """Background thread loop for delayed event processing.

        Monitors delay queue for events that have reached their delay time
        and processes them by triggering pipeline cycles.
        """
        while self._delay_thread_running:
            try:
                # Check the oldest queued event without removing it
                timestamp, data = self._delay_thread_queue.queue[0]
                now = time.monotonic()

                if now - timestamp >= self.source_delay:
                    # Delay period has elapsed, process the event
                    _, data = self._delay_thread_queue.get()
                    self._data = data
                    self.cycle()  # Trigger node cycle
                    self._data = None  # Clear data after processing
                else:
                    # Delay period not yet elapsed, wait briefly
                    time.sleep(0.001)
            except IndexError:
                # Queue is empty, wait briefly before checking again
                time.sleep(0.001)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return current event data if available.

        Args:
            data: Input data dictionary (unused for source nodes).

        Returns:
            Dictionary containing event data if event is active, empty dict
            otherwise.
        """
        return self._data if self._data is not None else {}
