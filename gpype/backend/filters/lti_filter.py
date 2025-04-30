from ..core.io_node import IONode
from ...common.constants import Constants
from typing import Dict
import numpy as np
from scipy.signal import lfilter, lfilter_zi

PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT


class LTIFilter(IONode):

    class Configuration(IONode.Configuration):
        class Keys(IONode.Configuration.Keys):
            B = 'b'
            A = 'a'

    def __init__(self,
                 b: np.ndarray = None,
                 a: np.ndarray = None,
                 **kwargs):
        if b is None:
            b = np.array([1])
        if a is None:
            a = np.array([1])
        super().__init__(b=b, a=a, **kwargs)
        self._z = None  # Initialize filter state

    def setup(self,
              data: Dict[str, np.ndarray],
              port_metadata_in: Dict[str, dict]) -> Dict[str, dict]:

        # Retrieve metadata and initialize the filter state
        md = port_metadata_in[PORT_IN]
        channel_count = md.get(Constants.Keys.CHANNEL_COUNT)
        if channel_count is None:
            raise ValueError("Channel count must be provided in metadata.")

        b = self.config[LTIFilter.Configuration.Keys.B]
        a = self.config[LTIFilter.Configuration.Keys.A]

        # Ensure the coefficients are valid
        if len(a) < 1 or a[0] == 0:
            raise ValueError("Invalid 'a' coefficients: first element must be "
                             "non-zero.")

        # Initialize filter state for each channel
        zi = lfilter_zi(b, a)  # Initial condition for the filter
        self._z = np.tile(zi, (channel_count, 1)).T

        return super().setup(data, port_metadata_in)

    def step(self,
             data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:  # noqa: E501
        data_in = data[PORT_IN]

        # Retrieve coefficients
        b = self.config[LTIFilter.Configuration.Keys.B]
        a = self.config[LTIFilter.Configuration.Keys.A]

        # Ensure input data is compatible
        if data_in.ndim != 2:
            raise ValueError("Input data must be a 2D array (samples x "
                             "channels).")

        # Apply the filter to each channel
        data_out, self._z = lfilter(b, a, data_in, axis=0, zi=self._z)

        return {PORT_OUT: data_out}
