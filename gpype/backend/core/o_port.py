import ioiocore as ioc
from ...common.constants import Constants
import numpy as np


class OPort(ioc.OPort):

    class Configuration(ioc.OPort.Configuration):

        class Keys(ioc.OPort.Configuration.Keys):
            pass

    def __init__(self,
                 name: str = Constants.Defaults.PORT_OUT,
                 timing: Constants.Timing = Constants.Timing.SYNC,
                 **kwargs):
        type_key = self.Configuration.Keys.TYPE
        type: str = kwargs.pop(type_key,
                               np.ndarray.__name__)  # noqa: E501

        super().__init__(name=name,
                         type=type,
                         timing=timing,
                         **kwargs)
