from pynput import keyboard
from .base.event_source import EventSource
from ...common.constants import Constants
import threading
import numpy as np

PORT_OUT = Constants.Defaults.PORT_OUT


class KeyboardCapture(EventSource):

    FINGERPRINT = "a9dcf27822847e6f6e0a22fa7f68fdcd"

    class Configuration(EventSource.Configuration):
        class Keys(EventSource.Configuration.Keys):
            pass

    def __init__(self, **kwargs):
        EventSource.__init__(self, **kwargs)
        self._running = False
        self._listener = None

    def trigger(self, data):
        super().trigger({PORT_OUT: np.array([[data]],
                                            dtype=Constants.DATA_TYPE)})

    def _on_press(self, key):
        try:
            key_value = key.value.vk
        except AttributeError:
            try:
                key_value = key.vk
            except AttributeError:
                key_value = -1

        self.trigger(key_value)  # Pass the key to trigger

    def _on_release(self, key):
        self.trigger(0)

    def start(self):
        EventSource.start(self)
        if not self._running:
            self._running = True
            self._press_listener = keyboard.Listener(on_press=self._on_press)
            self._press_listener.start()
            self._release_listener = keyboard.Listener(on_release=self._on_release)  # noqa: E501
            self._release_listener.start()

    def _start_thread_fun(self):
        self.trigger(np.array([[0]]))

    def stop(self):
        EventSource.stop(self)
        if self._running and self._press_listener:
            self._running = False
            self._press_listener.stop()
            self._press_listener.join()
            self._release_listener.stop()
            self._release_listener.join()
