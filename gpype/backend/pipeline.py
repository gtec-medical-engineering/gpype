import ioiocore as ioc
import os
import sys
from .core.node import Node

if sys.platform == "win32":
    LOG_DIR = os.path.join(os.getenv("APPDATA"), "gtec", "gPype")
elif sys.platform == "darwin":
    LOG_DIR = os.path.join(os.path.expanduser("~/Library/Application Support"),
                           "gtec",
                           "gPype")
else:
    raise NotImplementedError("Platform not supported.")


class Pipeline(ioc.Pipeline):

    def __init__(self):
        super().__init__(directory=LOG_DIR)

    def add_node(self, node: Node):
        super().add_node(node)

    def connect(self,
                output_node: ioc.ONode,
                input_node: ioc.INode):
        super().connect(output_node, input_node)

    def connect_ports(self,
                      output_node: ioc.ONode,
                      output_node_port: str,
                      input_node: ioc.INode,
                      input_node_port: str):
        super().connect_ports(output_node, output_node_port,
                              input_node, input_node_port)

    def start(self):
        super().start()

    def stop(self):
        super().stop()

    def serialize(self) -> dict:
        return super().serialize()
