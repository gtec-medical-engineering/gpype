from typing import Dict
from abc import abstractmethod
import numpy as np
import ioiocore as ioc
from ...common.constants import Constants
from .node import Node
from .o_port import OPort


class ONode(ioc.ONode, Node):

    def __init__(self,
                 output_ports: list[OPort.Configuration] = None,
                 **kwargs):
        ioc.ONode.__init__(self,
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
