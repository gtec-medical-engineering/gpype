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

    This class implements an event-driven source that generates data in
    response to external triggers rather than continuous sampling. It
    supports both immediate event processing and delayed event processing
    for timing control.

    The EventSource is particularly useful for:
    - Stimulus presentation systems
    - User interaction events
    - External trigger processing
    - Marker and annotation generation

    Features:
    - Asynchronous event triggering with configurable delay
    - Thread-safe event queuing for delayed processing
    - Automatic data type conversion and formatting
    - Integration with g.Pype timing framework

    Attributes:
        _data: Current data to be output in the next step
        _delay_thread_queue: Queue for delayed event processing
        _delay_thread: Background thread for delay management
        _delay_thread_running: Flag to control thread execution

    Note:
        Events can be triggered immediately or with a specified delay.
        The delay mechanism uses a separate thread to maintain real-time
        performance of the main processing pipeline.
    """

    def __init__(self, **kwargs):
        """Initialize the event source with asynchronous output configuration.

        Configures the event source with default asynchronous timing and
        initializes data storage for each configured output port. The source
        automatically handles port configuration and data array allocation.

        Args:
            **kwargs: Additional arguments passed to parent Source class.
                Can include channel_count, frame_size, and other source
                configuration parameters.

        Note:
            Output ports are automatically configured with ASYNC timing
            if not explicitly provided. Data arrays are pre-allocated
            for all non-inherited channel configurations.
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
        """Start the event source and initialize delay processing if needed.

        Initializes the event source by calling the parent start method,
        setting the running state, and starting the delay processing thread
        if source_delay is configured.

        Note:
            If source_delay > 0, a background thread is started to handle
            delayed event processing. The initial cycle() call ensures the
            source is ready for immediate event processing.
        """
        # Call parent start method first
        Source.start(self)

        # Set running state
        self.status = Constants.States.RUNNING

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
        """Stop the event source and clean up delay processing thread.

        Sets the stopped state and signals the delay processing thread
        to terminate. The parent stop method handles the main cleanup.

        Note:
            The delay thread is set to daemon mode, so it will automatically
            terminate when the main thread exits, but we explicitly signal
            it to stop for clean shutdown.
        """
        # Set stopped state
        self.status = Constants.States.STOPPED

        # Signal delay thread to stop
        if hasattr(self, "_delay_thread_running"):
            self._delay_thread_running = False

        # Call parent stop method
        Source.stop(self)

    def trigger(self, value):
        """Trigger an event with the specified value.

        Creates an event with the given value and either processes it
        immediately or queues it for delayed processing based on the
        configured source_delay.

        Args:
            value: The event value to be transmitted. Will be converted
                to the appropriate data type and formatted as a single-sample
                array for output.

        Note:
            If source_delay > 0, the event is queued with a timestamp for
            delayed processing. Otherwise, it's processed immediately with
            a cycle() call.
        """
        # Create data array with the event value
        data = {PORT_OUT: np.array([[value]], dtype=Constants.DATA_TYPE)}

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

        Continuously monitors the delay queue for events that have reached
        their delay time. When an event's delay period has elapsed, it
        processes the event by triggering a pipeline cycle.

        This method runs in a separate daemon thread to avoid blocking
        the main processing pipeline while waiting for delay periods
        to complete.

        Note:
            Uses brief sleep periods (1ms) to prevent excessive CPU usage
            while maintaining responsive timing. The thread terminates when
            _delay_thread_running is set to False.
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

        This method is called by the pipeline framework to retrieve data
        from the event source. It returns the current event data if an
        event has been triggered, or an empty dictionary if no event
        is pending.

        Args:
            data: Input data dictionary (unused for source nodes).

        Returns:
            Dictionary containing event data if an event is active,
            empty dictionary otherwise.

        Note:
            The returned data is only available for one step cycle after
            an event is triggered. This ensures events are processed
            exactly once by the pipeline.
        """
        return self._data if self._data is not None else {}
