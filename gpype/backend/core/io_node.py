import ioiocore as ioc
from abc import abstractmethod
from typing import Dict
from ...common.constants import Constants
from .node import Node
import numpy as np


class IONode(ioc.IONode, Node):

    def __init__(self,
                 input_ports: list[ioc.IPort.Configuration] = None,
                 output_ports: list[ioc.OPort.Configuration] = None,
                 **kwargs):
        ioc.IONode.__init__(self,
                            input_ports=input_ports,
                            output_ports=output_ports,
                            **kwargs)
        Node.__init__(self, target=self)

    def setup(self,
              data: Dict[str, np.ndarray],
              port_metadata_in: Dict[str, dict]) -> Dict[str, dict]:
        return super().setup(data, port_metadata_in)

    @abstractmethod
    def step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:  # noqa: E501
        pass
