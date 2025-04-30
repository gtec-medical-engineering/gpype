from typing import Dict
from abc import abstractmethod
import ioiocore as ioc
from ...common.constants import Constants
from .node import Node
from .i_port import IPort
import numpy as np


class INode(ioc.INode, Node):

    def __init__(self,
                 input_ports: list[IPort.Configuration] = None,
                 **kwargs):
        ioc.INode.__init__(self,
                           input_ports=input_ports,
                           **kwargs)
        Node.__init__(self, target=self)

    def setup(self,
              data: Dict[str, np.ndarray],
              port_metadata_in: Dict[str, dict]) -> Dict[str, dict]:
        return super().setup(data, port_metadata_in)

    @abstractmethod
    def step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:  # noqa: E501
        pass
