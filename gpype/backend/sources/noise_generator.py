import numpy as np
from typing import Dict
import ioiocore as ioc
from .base.fixed_rate_source import FixedRateSource
from ...utilities.constants import Constants
from ..core.o_port import OPort

OUT_PORT = ioc.Constants.Defaults.PORT_OUT


class NoiseGenerator(FixedRateSource):

    FINGERPRINT = "458da03c19bc53abd5a2f946cc6e7be5"

    class Configuration(FixedRateSource.Configuration):
        class Keys(FixedRateSource.Configuration.Keys):
            VARIANCE = 'variance'

    def __init__(self,
                 sampling_rate: float,
                 channel_count: int,
                 variance: float = 1,
                 **kwargs):
        if variance <= 0:
            raise ValueError("rate must be greater than zero.")
        output_ports = kwargs.pop(self.Configuration.Keys.OUTPUT_PORTS,
                                  [OPort.Configuration()])
        FixedRateSource.__init__(self,
                                 sampling_rate=sampling_rate,
                                 channel_count=channel_count,
                                 variance=variance,
                                 output_ports=output_ports,
                                 **kwargs)

    def step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:  # noqa: E501
        dims = (1, self.config[self.Configuration.Keys.CHANNEL_COUNT][0])
        std = Constants.DATA_TYPE(np.sqrt(self.config[self.Configuration.Keys.VARIANCE]))  # noqa: E501
        rng = np.random.default_rng()
        val = rng.standard_normal(size=dims,
                                  dtype=Constants.DATA_TYPE) * std
        return {OUT_PORT: val}
