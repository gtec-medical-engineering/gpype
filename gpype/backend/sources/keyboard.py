from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from ...common.constants import Constants
from .base.event_source import EventSource

# Port identifier for keyboard event output
PORT_OUT = Constants.Defaults.PORT_OUT


class Keyboard(EventSource):
    """
    Keyboard input source for capturing key press and release events.

    This class provides real-time keyboard event capture for BCI applications,
    enabling users to trigger events, control experiments, or provide input
    through keyboard interactions. It monitors both key press and release
    events, converting them to numerical values for pipeline processing.

    The keyboard source runs in the background and triggers events whenever
    keys are pressed or released. Key press events generate the virtual key
    code, while key release events generate a value of 0.

    Features:
        - Real-time keyboard event monitoring
        - Support for both printable and special keys
        - Press and release event detection

    Note:
        Requires appropriate permissions for global keyboard monitoring.
        May need to run with elevated privileges on some systems.
    """

    # Source code fingerprint
    FINGERPRINT = "0783b0f9b30cd1d85468d1f59cc15bec"

    class Configuration(EventSource.Configuration):
        """Configuration class for Keyboard source parameters."""

        class Keys(EventSource.Configuration.Keys):
            """Configuration key constants for the Keyboard source."""

            pass

    def __init__(self, **kwargs):
        """
        Initialize the keyboard event source.

        Args:
            **kwargs: Additional configuration parameters passed to
                EventSource base class.
        """
        # Initialize parent EventSource
        EventSource.__init__(self, **kwargs)

        # Initialize keyboard monitoring state
        self._running = False
        self._press_listener = None
        self._release_listener = None

    def _on_press(self, key):
        """
        Handle keyboard key press events.

        This callback is invoked when any key is pressed. It extracts the
        virtual key code and triggers an event in the BCI pipeline.

        Args:
            key: The pressed key object from pynput library.
                Can be KeyCode (printable keys) or Key (special keys).
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
        """
        Handle keyboard key release events.

        This callback is invoked when any key is released. It triggers
        an event with value 0 to indicate key release.

        Args:
            key: The released key object from pynput library.
        """
        # Always trigger 0 for key release events
        self.trigger(0)

    def start(self):
        """
        Start keyboard event monitoring.

        Initializes and starts the keyboard listeners for both press and
        release events. The listeners run in background threads to avoid
        blocking the main application.
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
        """
        Stop keyboard event monitoring and cleanup resources.

        Stops the keyboard listeners and waits for their threads to complete.
        This ensures clean shutdown and proper resource cleanup.
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
