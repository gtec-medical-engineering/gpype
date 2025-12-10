from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from ...common.constants import Constants
from .base.event_source import EventSource

# Port identifier for keyboard event output
PORT_OUT = Constants.Defaults.PORT_OUT


class Keyboard(EventSource):
    """Keyboard input source for capturing key press and release events.

    Provides real-time keyboard event capture. Monitors key press/release
    events and convertings them to numerical values. Key press events
    generate virtual key codes, release events generate 0.
    """

    # Source code fingerprint
    FINGERPRINT = "c8361f4118ae0a38530b422f8a20816c"

    class Configuration(EventSource.Configuration):
        """Configuration class for Keyboard source parameters."""

        class Keys(EventSource.Configuration.Keys):
            """Configuration key constants for the Keyboard source."""

            pass

    def __init__(self, **kwargs):
        """Initialize keyboard event source.

        Args:
            **kwargs: Additional configuration parameters for EventSource.
        """
        # Initialize parent EventSource
        EventSource.__init__(self, **kwargs)

        # Initialize keyboard monitoring state
        self._running = False
        self._press_listener = None
        self._release_listener = None

    def _on_press(self, key):
        """Handle keyboard key press events.

        Extracts virtual key code and triggers an event.

        Args:
            key: Pressed key object from pynput (KeyCode or Key).
        """
        # Extract virtual key code based on key type
        if isinstance(key, KeyCode):  # Printable keys (letters, digits, etc.)
            key_value = key.vk
        elif isinstance(key, Key):  # Special keys (ctrl, arrows, etc.)
            # Handle special keys with virtual key codes
            key_value = key.value.vk if hasattr(key.value, "vk") else -1
        else:
            # Unknown key type, use default value
            key_value = -1

        # Trigger event with the key code
        self.trigger(key_value)

    def _on_release(self, key):
        """Handle keyboard key release events.

        Triggers an event with value 0 to indicate key release.

        Args:
            key: Released key object from pynput.
        """
        # Always trigger 0 for key release events
        self.trigger(0)

    def start(self):
        """Start keyboard event monitoring.

        Initializes and starts keyboard listeners for press and release events
        in background threads.
        """
        # Only start if not already running
        if not self._running:
            self._running = True

            # Create and start key press listener
            self._press_listener = keyboard.Listener(on_press=self._on_press)
            self._press_listener.start()

            # Create and start key release listener
            self._release_listener = keyboard.Listener(
                on_release=self._on_release
            )
            self._release_listener.start()

        # Start parent EventSource
        EventSource.start(self)

    def stop(self):
        """Stop keyboard event monitoring and cleanup resources.

        Stops keyboard listeners and waits for their threads to complete.
        """
        # Stop parent EventSource first
        EventSource.stop(self)

        # Stop keyboard listeners if running
        if self._running:
            self._running = False

            # Stop and wait for press listener
            if self._press_listener:
                self._press_listener.stop()
                self._press_listener.join()
                self._press_listener = None

            # Stop and wait for release listener
            if self._release_listener:
                self._release_listener.stop()
                self._release_listener.join()
                self._release_listener = None
