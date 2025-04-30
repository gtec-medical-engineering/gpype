from ..core.io_node import IONode
from ...common.constants import Constants
from typing import Dict
import numpy as np
from scipy.signal import lfilter, lfilter_zi

PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT
EPS = 1e-10


class SQEstimator(IONode):

    class Configuration(IONode.Configuration):
        class Keys(IONode.Configuration.Keys):
            pass

    def __init__(self,
                 **kwargs):
        # Validate the coefficients
        super().__init__(**kwargs)
        self._z = None  # Initialize filter state
        self._a_buf = None
        self._b_buf = None
        self._a_hp = None
        self._b_hp = None
        self._a_lp = None
        self._b_lp = None
        self._target_lp = 12
        self._target_hp = 8
        self._margin_lp = 12
        self._margin_hp = 12

    def setup(self,
              data: Dict[str, np.ndarray],
              port_metadata_in: Dict[str, dict]) -> Dict[str, dict]:

        # Retrieve metadata and initialize the filter state
        md = port_metadata_in[PORT_IN]
        channel_count = md.get(Constants.Keys.CHANNEL_COUNT)
        if channel_count is None:
            raise ValueError("Channel count must be provided in metadata.")

        sampling_rate = md.get(Constants.Keys.SAMPLING_RATE)
        if sampling_rate is None:
            raise ValueError("Sampling rate count must be provided in metadata.")  # noqa: E501

        from scipy.signal import butter
        order = 2
        f_lo1 = 0.5
        f_lo2 = 10
        f_hi = 15
        self._b_lp, self._a_lp = butter(order, np.array([f_lo1, f_lo2]) / (sampling_rate / 2), btype='bandpass')

        # Low-pass filter
        self._b_hp, self._a_hp = butter(order, f_hi / sampling_rate * 2, btype='highpass')

        B = np.uint(np.round(sampling_rate / 2))
        # self._b_buf_hp = [1 / B] * B
        # self._a_buf_hp = [1]
        # self._b_buf_lp = [1 / B] * B
        # self._a_buf_lp = [1]

        alpha = 0.985
        self._b_buf_hp = [1 - alpha]
        self._a_buf_hp = [1, -alpha]
        self._b_buf_lp = [1 - alpha]
        self._a_buf_lp = [1, -alpha]


        # a = self.config[LTIFilter.Configuration.Keys.A]

        # Ensure the coefficients are valid
        # if len(a) < 1 or a[0] == 0:
        #     raise ValueError("Invalid 'a' coefficients: first element must be "
        #                      "non-zero.")

        # Initialize filter state for each channel
        zi_buf_hp = lfilter_zi(self._b_buf_hp, self._a_buf_hp)  # Initial condition for the filter
        self._z_buf_hp = np.tile(zi_buf_hp, (channel_count, 1)).T
        zi_buf_lp = lfilter_zi(self._b_buf_lp, self._a_buf_lp)  # Initial condition for the filter
        self._z_buf_lp = np.tile(zi_buf_lp, (channel_count, 1)).T
        zi_hp = lfilter_zi(self._b_hp, self._a_hp)  # Initial condition for the filter
        self._z_hp = np.tile(zi_hp, (channel_count, 1)).T
        zi_lp = lfilter_zi(self._b_lp, self._a_lp)  # Initial condition for the filter
        self._z_lp = np.tile(zi_lp, (channel_count, 1)).T

        return super().setup(data, port_metadata_in)

    def step(self,
             data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:  # noqa: E501
        data_in = data[PORT_IN]

        # Ensure input data is compatible
        if data_in.ndim != 2:
            raise ValueError("Input data must be a 2D array (channels x "
                             "samples).")

        # Apply the filter to each channel
        x_hp, self._z_hp = lfilter(self._b_hp,
                                   self._a_hp,
                                   data_in,
                                   axis=0,
                                   zi=self._z_hp)
        x_lp, self._z_lp = lfilter(self._b_lp,
                                   self._a_lp,
                                   data_in,
                                   axis=0,
                                   zi=self._z_lp)
        x_buf_lp, self._z_buf_lp = lfilter(self._b_buf_lp,
                                           self._a_buf_lp,
                                           np.square(x_lp),
                                           axis=0,
                                           zi=self._z_buf_lp)
        x_buf_hp, self._z_buf_hp = lfilter(self._b_buf_hp,
                                           self._a_buf_hp,
                                           np.square(x_hp),
                                           axis=0,
                                           zi=self._z_buf_hp)

        sq_lp = 10 * np.log10(x_buf_lp + EPS)
        sq_lp = (sq_lp - self._target_lp) / self._margin_lp
        sq_lp = abs(sq_lp)
        sq_hp = 10 * np.log10(x_buf_hp + EPS)
        sq_hp = (sq_hp - self._target_hp) / self._margin_hp
        sq_hp = abs(sq_hp)

        sq = np.maximum(sq_hp, sq_lp)

        return {PORT_OUT: sq}
