import ioiocore as ioc
from ...common.constants import Constants
import numpy as np


class IPort(ioc.IPort):

    class Configuration(ioc.IPort.Configuration):

        class Keys(ioc.IPort.Configuration.Keys):
            pass

    def __init__(self,
                 name: str = Constants.Defaults.PORT_IN,
                 timing: Constants.Timing = Constants.Timing.SYNC,
                 **kwargs):
        type_key = self.Configuration.Keys.TYPE
        type: str = kwargs.pop(type_key,
                               np.ndarray.__name__)  # noqa: E501

        super().__init__(name=name,
                         type=type,
                         timing=timing,
                         **kwargs)
