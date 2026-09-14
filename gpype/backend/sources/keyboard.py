from typing import List

import ioiocore as ioc

from ...common.constants import Constants
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core._private.sync import Sync
from ..core.o_port import OPort
from .base import raw
from .base.event_source import EventSource

# Port identifier for keyboard event output
PORT_OUT = Constants.Defaults.PORT_OUT


#: The pynput names, or None until something needs them. Same reasoning
#: as BCICore8's: a document naming Keyboard is deserialised in every
#: residency, but the listener only runs where somebody is at the
#: keyboard. pynput additionally raises at import on Linux with no
#: display, so importing eagerly would make a headless process refuse a
#: document it was never going to read keys in.
keyboard = None
Key = None
KeyCode = None


def _load() -> None:
    """Import pynput.keyboard, once, on first use.

    Only fills names that are still None, so a test that has replaced
    them with stand-ins keeps them.
    """
    global keyboard, Key, KeyCode
    if keyboard is None:
        from pynput import keyboard as module

        keyboard = module
    if Key is None:
        Key = keyboard.Key
    if KeyCode is None:
        KeyCode = keyboard.KeyCode


class _KeyboardCore(EventSource):
    """Internal node implementing keyboard event capture logic.

    This is the actual keyboard node (pure ONode inheritance via EventSource).
    It is wrapped by the Keyboard chain for distributed operation.

    Provides real-time keyboard event capture. Monitors key press/release
    events and converts them to numerical values. Key press events
    generate virtual key codes, release events generate 0.
    """

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
        # The library is imported here, not at module scope: this core is
        # built only where the hardware is, while the chain that names it
        # is built in every residency.
        _load()
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


class Keyboard(ioc.OChain):
    """Keyboard input chain for capturing key press and release events.

    This is an OChain that contains:
    - _KeyboardCore: The actual keyboard event capture node
    - Link: Bridge for distributed operation (passthrough in standalone)

    The chain structure enables distributed edge/server operation while
    keeping node inheritance clean (no chain mixing in node path).

    Provides real-time keyboard event capture. Monitors key press/release
    events and converts them to numerical values.
    """

    def __init__(self, **kwargs):
        """Initialize keyboard event chain.

        Args:
            **kwargs: Additional configuration parameters.
        """
        # Store parameters for create_internal_nodes
        self._link_stream_id = stream_id_for(kwargs)
        self._core_params = strip_chain_keys(kwargs)

        # Initialize OChain (calls create_internal_nodes)
        kwargs.setdefault(
            self.Configuration.Keys.OUTPUT_PORTS,
            [OPort.Configuration()],
        )
        ioc.OChain.__init__(
            self,
            stream_id=self._link_stream_id,
            **kwargs,
        )

    def create_internal_nodes(self) -> List[ioc.Node]:
        """Create the internal node chain.

        Returns:
            The head of the list is whatever stands in for this chain's
            core: the core itself, the core followed by a raw tap
            under ``save_as``, a replay core under ``load_from``,
            or nothing at all under SERVER residency. Then a Link
            where the pipeline is distributed, and Sync last.
        """
        from ...common.launch_config import LaunchConfig

        nodes = []
        residency = LaunchConfig.get().residency
        # Recorded, replaced, or simply built, as the launch
        # configuration says. Contributes nothing under server
        # residency: the core lives on the edge, and so does
        # anything recording or replaying it.
        nodes.extend(
            raw.source_stage(
                self,
                lambda: _KeyboardCore(**self._core_params),
                timing=Constants.Timing.ASYNC,
            )
        )
        if residency != Constants.Residency.STANDALONE:
            nodes.append(
                Link(
                    sender=Constants.Residency.EDGE,
                    receiver=Constants.Residency.SERVER,
                    stream_id=self._link_stream_id,
                    # An event stream is sparse on both sides of the
                    # link, so the ports have to say so. Without this the
                    # ASYNC output of the event core meets a SYNC input
                    # and the chain cannot be built at all.
                    timing=Constants.Timing.ASYNC,
                )
            )
        # Sync places each event on the master timeline. ASYNC
        # because a sparse stream must stay sparse: announced as
        # continuous, every downstream node would wait for data on
        # every cycle.
        nodes.append(Sync(timing=Constants.Timing.ASYNC))
        return nodes
