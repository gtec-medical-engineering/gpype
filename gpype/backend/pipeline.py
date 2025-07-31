from typing import Union

import ioiocore as ioc

from .core.node import Node


class Pipeline(ioc.Pipeline):
    """
    Brain-Computer Interface pipeline for real-time data processing.

    The Pipeline class extends the ioiocore Pipeline to provide specialized
    functionality for BCI applications. It manages the execution of
    interconnected nodes that process neural signals in real-time.

    The pipeline handles node lifecycle management, data flow connections,
    and provides logging capabilities for debugging and analysis. It serves
    as the central orchestrator for all BCI processing components.

    Features:
        - Real-time node execution and data flow
        - Automatic logging to platform-specific directories
        - Node connection management with port-based routing
        - Start/stop lifecycle control
        - Pipeline serialization

    Note:
        The pipeline automatically creates log files in platform-appropriate
        directories for debugging and performance analysis.
    """

    def __init__(self):
        """
        Initialize the Pipeline with platform-specific logging directory.

        Creates a new pipeline instance configured for BCI data processing
        with automatic logging enabled in the appropriate system directory.
        """
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
        """
        Connect two nodes to establish data flow in the pipeline.

        Creates a connection between a source node and target node, allowing
        data to flow from source outputs to target inputs. Supports both
        simple node-to-node connections and specific port connections using
        dictionary notation.

        Args:
            source (Union[Node, dict]): Source node or port specification.
                - Node: Connect default output port to target
                - dict: Specify source node and port (e.g., node["port_name"])
            target (Union[Node, dict]): Target node or port specification.
                - Node: Connect to default input port
                - dict: Specify target node and port (e.g., node["port_name"])

        Notes:
            Port names depend on the specific node implementation.
                Common port names include "in", "out", "trigger".
            When connecting nodes, these nodes are implicitly added to the
                pipeline if not already present.
        """
        super().connect(source, target)

    def start(self):
        """
        Start the pipeline and begin real-time data processing.

        Initiates the execution of all nodes in the pipeline according to
        their configured connections and timing. The pipeline will run
        continuously until stop() is called.

        The start process includes:
        - Node initialization and setup
        - Port connection validation
        - Threading setup for real-time execution
        - Data flow activation

        Note:
            This method is non-blocking.
        """
        super().start()

    def stop(self):
        """
        Stop the pipeline and terminate all data processing.

        Gracefully shuts down the pipeline by stopping all nodes and
        cleaning up resources. This ensures proper cleanup of threads,
        file handles, hardware connections, and other resources.

        The stop process includes:
        - Signal termination to all nodes
        - Thread cleanup and synchronization
        - Resource deallocation
        - Connection cleanup

        Note:
            Always call stop() before program termination to ensure
            proper cleanup, especially when using hardware interfaces.
        """
        super().stop()

    def serialize(self) -> dict:
        """
        Serialize the pipeline configuration to a dictionary.

        Converts the current pipeline state, including all nodes and their
        connections, into a dictionary format suitable for storage or
        transmission. This enables pipeline configuration persistence
        and reconstruction.

        Returns:
            dict: Dictionary containing the complete pipeline configuration,
                including nodes, connections, parameters, and metadata.

        Note:
            The serialized configuration can be used to reconstruct an
            identical pipeline with the same nodes and connections.
        """
        return super().serialize()

    @staticmethod
    def deserialize(data: dict) -> "Pipeline":
        """
        Deserialize a pipeline configuration from a dictionary.

        Reconstructs a Pipeline instance from a serialized configuration
        dictionary. This allows for loading previously saved pipeline states
        and restoring node connections and parameters.

        Args:
            data (dict): Serialized pipeline configuration dictionary.

        Returns:
            Pipeline: A new Pipeline instance with the specified configuration.
        """
        return super().deserialize(data)
