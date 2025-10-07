from typing import Union

import ioiocore as ioc

from .core.node import Node


class Pipeline(ioc.Pipeline):
    """Brain-Computer Interface pipeline for real-time data processing.

    Extends ioiocore Pipeline for BCI applications with automatic logging
    to platform-specific directories. Manages node lifecycle, data flow
    connections, and real-time execution of interconnected processing nodes.
    """

    def __init__(self):
        """Initialize Pipeline with platform-specific logging directory."""
        # Determine platform-specific log directory
        import os
        import sys

        if sys.platform == "win32":
            log_dir = os.path.join(os.getenv("APPDATA", ""), "gtec", "gPype")
        elif sys.platform == "darwin":
            app_support = os.path.expanduser("~/Library/Application Support")
            log_dir = os.path.join(app_support, "gtec", "gPype")
        else:
            log_dir = None  # Use default ioiocore directory

        # Initialize parent pipeline with logging directory
        super().__init__(directory=log_dir)

    def connect(self, source: Union[Node, dict], target: Union[Node, dict]):
        """Connect two nodes to establish data flow in the pipeline.

        Args:
            source (Union[Node, dict]): Source node or port specification.
                Use dict for specific ports (e.g., node["port_name"]).
            target (Union[Node, dict]): Target node or port specification.
                Use dict for specific ports (e.g., node["port_name"]).

        Note:
            Nodes are automatically added to pipeline if not already present.
        """
        super().connect(source, target)

    def start(self):
        """Start the pipeline and begin real-time data processing.

        Initiates execution of all nodes according to their configured
        connections and timing. Runs continuously until stop() is called.

        Note:
            This method is non-blocking.
        """
        super().start()

    def stop(self):
        """Stop the pipeline and terminate all data processing.

        Gracefully shuts down all nodes and cleans up resources including
        threads, file handles, and hardware connections.

        Note:
            Always call stop() before program termination to ensure
            proper cleanup, especially when using hardware interfaces.
        """
        super().stop()

    def serialize(self) -> dict:
        """Serialize the pipeline configuration to a dictionary.

        Returns:
            dict: Dictionary containing the complete pipeline configuration,
                including nodes, connections, parameters, and metadata.
        """
        return super().serialize()

    @staticmethod
    def deserialize(data: dict) -> "Pipeline":
        """Deserialize a pipeline configuration from a dictionary.

        Args:
            data (dict): Serialized pipeline configuration dictionary.

        Returns:
            Pipeline: A new Pipeline instance with the specified configuration.
        """
        return super().deserialize(data)
