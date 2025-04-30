import ioiocore as ioc
from ...backend.core.i_port import IPort
from typing import Dict
import numpy as np
from ...common.constants import Constants
from ...backend.core.i_node import INode
import queue
import threading
from datetime import datetime
import os


class FileSink(INode):

    class Configuration(ioc.INode.Configuration):
        class Keys(ioc.INode.Configuration.Keys):
            FILE_NAME = "file_name"

    def __init__(self,
                 file_name: str,
                 input_ports: list[IPort.Configuration] = None,
                 **kwargs):
        if input_ports is None:
            input_ports = [IPort.Configuration()]
        INode.__init__(self,
                       input_ports=input_ports,
                       file_name=file_name,
                       **kwargs)
        self._file_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._file_handle = None

    def start(self):
        file_name = self.config[FileSink.Configuration.Keys.FILE_NAME]
        name, ext = os.path.splitext(file_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{name}_{timestamp}{ext}"
        if not file_name.endswith(".csv"):
            raise ValueError(f"Unsupported file format {file_name}.")

        self._file_handle = open(file_name, "w")
        self._worker_thread = threading.Thread(target=self._file_worker,
                                               daemon=True)
        self._stop_event.clear()
        self._worker_thread.start()
        super().start()

    def stop(self):
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join()
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        super().stop()

    def setup(self,
              data: Dict[str, np.ndarray],
              port_metadata_in: Dict[str, dict]) -> Dict[str, dict]:
        if not self._file_handle:
            raise RuntimeError("File handle is not initialized.")

        channel_count = data[Constants.Defaults.PORT_IN].shape[1]
        header = "Index, "
        header += ", ".join([f"Ch{d + 1:02d}" for d in range(channel_count)])
        self._file_queue.put(header)
        return {}

    def step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:  # noqa: E501
        d = data[Constants.Defaults.PORT_IN]
        for n in range(d.shape[0]):
            self._file_queue.put(f"{self.get_counter()}, " + ", ".join(map(str, d[0, :].tolist())))  # noqa: E501
        return {}

    def _file_worker(self):
        while not self._stop_event.is_set() or not self._file_queue.empty():
            try:
                line = self._file_queue.get(timeout=1)
                if self._file_handle:
                    self._file_handle.write(line + "\n")
                    self._file_handle.flush()
            except queue.Empty:
                continue
