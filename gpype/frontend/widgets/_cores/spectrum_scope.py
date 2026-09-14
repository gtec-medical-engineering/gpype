"""The widget half of SpectrumScope: everything that needs Qt.

Imported by SpectrumScope.create_internal_nodes only when the residency
actually builds a display. Kept out of the chain's own module so that a
server process -- which never builds one -- does not import Qt in order
to deserialize a document that names a scope.
"""

import threading
import time

import ioiocore as ioc
import numpy as np
from PySide6.QtGui import QPalette

from ....backend.core.i_port import IPort
from ....common._private.naming import node_label
from ....common.constants import Constants
from .._validation import check_bounds
from ..base.scope import Scope
from ..spectrum_scope import SpectrumScope

#: Default input port identifier
PORT_IN = ioc.Constants.Defaults.PORT_IN


class _SpectrumScopeCore(Scope):
    """Internal scope node for frequency-domain spectral visualization.

    Displays real-time frequency spectrum of input signals with configurable
    amplitude limits and averaging. Wrapped by SpectrumScope for distributed
    operation.
    """

    #: Default maximum amplitude for display scaling
    DEFAULT_AMPLITUDE_LIMIT = SpectrumScope.DEFAULT_AMPLITUDE_LIMIT
    #: Default number of spectra to average for smoothing
    DEFAULT_NUM_AVERAGES = SpectrumScope.DEFAULT_NUM_AVERAGES

    class Configuration(Scope.Configuration):

        class Keys(Scope.Configuration.Keys):
            #: Configuration key for maximum amplitude display limit
            AMPLITUDE_LIMIT = "amplitude_limit"
            #: Configuration key for number of averaging samples
            NUM_AVERAGES = "num_averages"

        class EmptyableKeys:
            """Optional keys that may also be *empty*.

            Deliberately not ``OptionalKeys``. That name means one
            specific thing to ioiocore -- may be absent, but must not be
            empty when present (``Configuration.__init__``) -- and an
            empty list is a legitimate value for every key declared here:
            no markers, no hidden channels. Declaring them under the name
            the validator reads makes ``markers=[]`` raise.

            Measured, so that nobody has to rediscover it::

                OptionalKeys   markers=[]  -> ValueError, must not be empty
                EmptyableKeys  markers=[]  -> accepted

            The name is still the declaration a reader and a catalog
            generator look for; it is only kept out of the validator's
            emptiness rule. The underlying gap is ioiocore's: there is no
            way to say "optional, and empty is fine".
            """

            #: Configuration key for list of channels to hide from display
            HIDDEN_CHANNELS = "hidden_channels"

    def __init__(
        self,
        amplitude_limit: float = None,
        num_averages: int = None,
        hidden_channels: list = None,
        **kwargs,
    ):
        """Initialize the SpectrumScope widget.

        Args:
            amplitude_limit (float, optional): Maximum amplitude for display
                scaling. Defaults to 50.
            num_averages (int, optional): Number of spectra to average.
                Defaults to 10.
            hidden_channels (list, optional): List of channel indices to hide.
                Defaults to empty list.
            **kwargs: Additional arguments passed to parent classes.
        """

        if amplitude_limit is None:
            amplitude_limit = self.DEFAULT_AMPLITUDE_LIMIT

        if num_averages is None:
            num_averages = self.DEFAULT_NUM_AVERAGES

        # See the chain: one bound, shared.
        check_bounds(
            "amplitude_limit",
            amplitude_limit,
            SpectrumScope.MIN_AMPLITUDE_LIMIT,
            SpectrumScope.MAX_AMPLITUDE_LIMIT,
            unit="uV",
        )

        if hidden_channels is None:
            hidden_channels = []

        # Take both from kwargs when present: a scope rebuilt from a
        # stored configuration supplies them there, and passing them
        # here as well would be a duplicate keyword.
        keys = self.Configuration.Keys
        input_ports = kwargs.pop(
            keys.INPUT_PORTS, [IPort.Configuration(name=PORT_IN)]
        )
        kwargs.setdefault(keys.NAME, "Spectrum Scope")

        Scope.__init__(
            self,
            input_ports=input_ports,
            amplitude_limit=amplitude_limit,
            hidden_channels=hidden_channels,
            num_averages=num_averages,
            **kwargs,
        )
        #: Maximum number of data points for plotting
        self._max_points: int = None
        #: Buffer for storing raw FFT data
        self._data_buffer: np.ndarray = None
        #: Buffer for averaged display data
        self._display_buffer: np.ndarray = None
        #: Current plot buffer index
        self._plot_index: int = 0
        #: Flag indicating if buffer is completely filled
        self._buffer_full: bool = False
        #: Current sample index for data tracking
        self._sample_index: int = 0
        #: Timestamp when widget was initialized
        self._start_time = time.time()
        #: Counter for display update operations
        self._update_counts = 0
        #: Counter for data processing steps
        self._step_counts = 0
        #: Current step processing rate in Hz
        self._step_rate = 0
        #: Thread lock for data buffer synchronization
        self._lock = threading.Lock()
        #: Flag indicating new data is available for display
        self._new_data = False
        #: Label widget for displaying rate information
        self._rate_label = None
        p = self.widget.palette()
        #: Foreground color from system theme
        self._foreground_color = p.color(QPalette.ColorRole.WindowText)
        #: Background color from system theme
        self._background_color = p.color(QPalette.ColorRole.Window)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Set up the spectrum scope with frequency vector and channels.

        Args:
            data (dict): Initial data dictionary.
            port_context_in (dict): Input port context information.

        Returns:
            dict: Output port context from parent setup.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        c = port_context_in[PORT_IN]

        # The name the author typed: this class is the private core of
        # the SpectrumScope chain, so type(self).__name__ on its own
        # prints "_SpectrumScopeCore", a symbol the public API has not
        # got. self.name adds the instance once an author has named one.
        label = node_label(self)

        # Each of these is published by the node upstream, so the
        # message says which port was read and what it did carry.
        got = ", ".join(sorted(c)) or "an empty context"
        sampling_rate = c.get(Constants.Keys.SAMPLING_RATE)
        if sampling_rate is None:
            raise ValueError(
                f"sampling_rate must be provided in the context of "
                f"port '{PORT_IN}'; {label} got {got}. The scope needs "
                f"it to label the frequency axis. Connect {label} "
                f"downstream of a source."
            )
        channel_count = c.get(Constants.Keys.CHANNEL_COUNT)
        if channel_count is None:
            raise ValueError(
                f"channel_count must be provided in the context of "
                f"port '{PORT_IN}'; {label} got {got}. Connect {label} "
                f"downstream of a node that declares how many channels "
                f"it emits."
            )
        frame_size = c.get(Constants.Keys.FRAME_SIZE)
        if frame_size is None:
            raise ValueError(
                f"frame_size must be provided in the context of port "
                f"'{PORT_IN}'; {label} got {got}. Connect {label} "
                f"downstream of a node that declares its frame size."
            )
        if frame_size <= 1:
            raise ValueError(
                f"frame_size must be greater than 1; {label} got "
                f"{frame_size} on port '{PORT_IN}'. One sample has no "
                f"spectrum -- the FFT transforms a window, and its "
                f"resolution is the sampling rate divided by that "
                f"window. Set frame_size on the source, or insert a "
                f"Framer before {label} to collect samples into "
                f"windows."
            )
        self._f_vec = np.fft.rfftfreq(frame_size, 1 / sampling_rate)
        hidden_channels = self.config[
            self.Configuration.EmptyableKeys.HIDDEN_CHANNELS
        ]
        self._channel_vec = [
            i for i in range(channel_count) if i not in hidden_channels
        ]
        self._channel_count = len(self._channel_vec)
        self._channel_labels = self._resolve_channel_labels(c)
        self._frame_size = frame_size
        self._sampling_rate = sampling_rate
        self._data_buffer: list = []
        self._num_averages = self.config[self.Configuration.Keys.NUM_AVERAGES]
        self._display_buffer = np.zeros((frame_size, self._channel_count))
        self._new_data = False
        self._start_time = time.time()
        return super().setup(data, port_context_in)

    def _update(self):
        """Update the spectrum display with current averaged data.

        Called periodically by the widget timer to refresh the frequency
        domain visualization with averaged spectral data.
        """

        if not self._new_data:
            return

        # Set up UI elements. Note that this has to be done in the main Qt
        # thread (like this)
        ylim = (0, self._channel_count)
        if self._curves is None:

            # Create curves
            [self.add_curve() for _ in range(self._channel_count)]
            amp_lim = self.config[self.Configuration.Keys.AMPLITUDE_LIMIT]
            yl = f"EEG Amplitudes (0 ... {amp_lim} µV)"
            self.set_labels(x_label="Frequency (Hz)", y_label=yl)
            ticks = [
                (
                    self._channel_count - i - 0.5,
                    self._channel_label(self._channel_vec[i]),
                )
                for i in range(self._channel_count)
            ]
            self._plot_item.getAxis("left").setTicks([ticks])
            self._plot_item.setYRange(*ylim)

        with self._lock:
            if not self._data_buffer:
                return
            self._display_buffer = np.mean(
                np.stack(self._data_buffer, axis=2), axis=2
            )
            self._display_buffer = np.abs(self._display_buffer)
            self._new_data = False

        ch_lim_key = self.Configuration.Keys.AMPLITUDE_LIMIT
        ch_lim = self.config[ch_lim_key]
        for i in range(len(self._channel_vec)):
            d = self._channel_count - i - 0.5
            self._curves[i].setData(
                self._f_vec,
                self._display_buffer[:, self._channel_vec[i]] / ch_lim / 2 + d,
                antialias=False,
            )

        # update xlim
        fw = self._f_vec[-1]
        margin = fw * 0.0125
        xlim = (-margin, fw + margin)
        self._plot_item.setXRange(*xlim)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process incoming FFT data and update buffer for averaging.

        Args:
            data (dict): Dictionary containing FFT amplitude data.

        Returns:
            dict: Unchanged input data (pass-through).
        """
        with self._lock:
            fft_input = data[PORT_IN]
            self._data_buffer.append(fft_input)
            if len(self._data_buffer) > self._num_averages:
                self._data_buffer.pop(0)
        self._new_data = True
