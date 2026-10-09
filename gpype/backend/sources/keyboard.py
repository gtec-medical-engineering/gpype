from __future__ import annotations

import sys
from typing import Optional

from ...common.constants import Constants
from .base.event_source import EventSource

# Port identifier for keyboard event output
PORT_OUT = Constants.Defaults.PORT_OUT

#: The platform whose key numbering pynput reports. Windows reports
#: Windows virtual-key codes, which is the numbering this node emits
#: everywhere, so there it passes them through; elsewhere it translates.
#: A module attribute so a test can choose the platform it simulates.
_PLATFORM = sys.platform

#: pynput's backend, the module its classes come from: "win32", "darwin",
#: "xorg" or "uinput"; None until start() loads it. Only Linux reads it,
#: because its two backends report a bare character, one with no native
#: code, for different keys: xorg for the keypad, uinput for a digit.
#: A module attribute so a test can choose the backend it simulates.
_BACKEND = None

#: What a key the numbering has no code for emits, as before.
UNKNOWN = -1

#: Windows virtual-key code of every pynput special key, by member name.
#: The names are the same on every pynput backend; the values are what
#: pynput's own Windows backend gives them (``pynput.keyboard._win32``),
#: aliases included -- ``cmd_l`` and ``shift_l`` share their value with
#: ``cmd`` and ``shift`` there. ``media_eject`` exists only on macOS and
#: has no Windows code, so it emits UNKNOWN.
SPECIAL_KEYS = {
    "alt": 0x12,
    "alt_l": 0xA4,
    "alt_r": 0xA5,
    "alt_gr": 0xA5,
    "backspace": 0x08,
    "caps_lock": 0x14,
    "cmd": 0x5B,
    "cmd_l": 0x5B,
    "cmd_r": 0x5C,
    "ctrl": 0x11,
    "ctrl_l": 0xA2,
    "ctrl_r": 0xA3,
    "delete": 0x2E,
    "down": 0x28,
    "end": 0x23,
    "enter": 0x0D,
    "esc": 0x1B,
    **{f"f{n}": 0x6F + n for n in range(1, 25)},
    "home": 0x24,
    "left": 0x25,
    "page_down": 0x22,
    "page_up": 0x21,
    "right": 0x27,
    "shift": 0xA0,
    "shift_l": 0xA0,
    "shift_r": 0xA1,
    "space": 0x20,
    "tab": 0x09,
    "up": 0x26,
    "media_play_pause": 0xB3,
    "media_stop": 0xB2,
    "media_volume_mute": 0xAD,
    "media_volume_down": 0xAE,
    "media_volume_up": 0xAF,
    "media_previous": 0xB1,
    "media_next": 0xB0,
    "insert": 0x2D,
    "menu": 0x5D,
    "num_lock": 0x90,
    "pause": 0x13,
    "print_screen": 0x2C,
    "scroll_lock": 0x91,
}

#: SPECIAL_KEYS as a key is named off Windows. There pynput gives the
#: generic ``alt`` and ``ctrl`` the left key's own code, so its Enum makes
#: ``alt_l`` and ``ctrl_l`` their aliases and a left Alt is reported as
#: ``Key.alt`` (``_darwin.py``, ``_xorg.py``, ``_uinput.py``). On Windows
#: ``alt`` is the side-less VK_MENU, and a left Alt is VK_LMENU. ``shift``
#: and ``cmd`` are the left keys' codes on Windows already.
_OFF_WINDOWS = {
    **SPECIAL_KEYS,
    "alt": SPECIAL_KEYS["alt_l"],
    "ctrl": SPECIAL_KEYS["ctrl_l"],
}


def _character_codes() -> dict:
    """Windows virtual-key code of each character a US layout types.

    Windows numbers a character key by the key, not the character: ``a``
    and ``A`` are both 0x41, ``1`` and ``!`` both 0x31. Letters and
    digits are layout-independent there; punctuation takes the ``VK_OEM``
    code of the key a US layout puts it on, which is the only layout this
    table can know.

    Returns:
        ``{character: code}``.
    """
    codes = {}
    for letter in "abcdefghijklmnopqrstuvwxyz":
        codes[letter] = codes[letter.upper()] = ord(letter.upper())
    for digit, shifted in zip("0123456789", ")!@#$%^&*("):
        codes[digit] = codes[shifted] = ord(digit)
    for pair, code in (
        (";:", 0xBA),
        ("=+", 0xBB),
        (",<", 0xBC),
        ("-_", 0xBD),
        (".>", 0xBE),
        ("/?", 0xBF),
        ("`~", 0xC0),
        ("[{", 0xDB),
        ("\\|", 0xDC),
        ("]}", 0xDD),
        ("'\"", 0xDE),
    ):
        for character in pair:
            codes[character] = code
    codes[" "] = 0x20
    # Ctrl with a letter types a control character on macOS (Ctrl+A is
    # U+0001); Windows still reports the letter's key.
    for n in range(1, 27):
        codes[chr(n)] = 0x40 + n
    return codes


#: See _character_codes.
CHARACTERS = _character_codes()

#: The numeric keypad, by macOS key code (``kVK_ANSI_Keypad*``). Checked
#: before the character, because Windows numbers these keys apart from
#: the digits they type: keypad 1 is VK_NUMPAD1, 0x61, not 0x31.
_DARWIN_KEYPAD = {
    0x52: 0x60,
    0x53: 0x61,
    0x54: 0x62,
    0x55: 0x63,
    0x56: 0x64,
    0x57: 0x65,
    0x58: 0x66,
    0x59: 0x67,
    0x5B: 0x68,
    0x5C: 0x69,
    0x43: 0x6A,
    0x45: 0x6B,
    0x4E: 0x6D,
    0x41: 0x6E,
    0x4B: 0x6F,
    0x4C: 0x0D,
}

#: The same, by X keysym (``XK_KP_*``).
_XORG_KEYPAD = {
    **{0xFFB0 + n: 0x60 + n for n in range(10)},
    0xFFAA: 0x6A,
    0xFFAB: 0x6B,
    0xFFAC: 0x6C,
    0xFFAD: 0x6D,
    0xFFAE: 0x6E,
    0xFFAF: 0x6F,
    0xFF8D: 0x0D,
}

#: The keysym pynput's X listener folds a numlocked keypad key away from.
#: It reports keypad 1 as a bare "1", with no keysym, where every other
#: character carries its own (``Listener._KEYPAD_KEYS`` in
#: ``pynput.keyboard._xorg``). The decimal key is "," whatever the layout
#: types. Keypad "=", which a PC keypad lacks, reads as "=", as on macOS.
#:
#: Not for an injected key. pynput's Controller hands this process's
#: listeners each character it types as that bare character, marked
#: injected (``Listener._on_fake_event``), having typed the main key; a
#: press the X server recorded is never marked injected. The X keycode
#: would say which key it was, but the callback is given only the key.
_XORG_FOLDED_KEYPAD = {
    **{str(n): 0xFFB0 + n for n in range(10)},
    "*": 0xFFAA,
    "+": 0xFFAB,
    "-": 0xFFAD,
    ",": 0xFFAE,
    "/": 0xFFAF,
}

#: The letter and digit keys by macOS key code, for a key whose character
#: says nothing -- Option+A types "å". By position on a US layout, so on
#: another layout this is the key, not the letter printed on it.
_DARWIN_KEYS = {
    0x00: "A",
    0x01: "S",
    0x02: "D",
    0x03: "F",
    0x04: "H",
    0x05: "G",
    0x06: "Z",
    0x07: "X",
    0x08: "C",
    0x09: "V",
    0x0B: "B",
    0x0C: "Q",
    0x0D: "W",
    0x0E: "E",
    0x0F: "R",
    0x10: "Y",
    0x11: "T",
    0x12: "1",
    0x13: "2",
    0x14: "3",
    0x15: "4",
    0x16: "6",
    0x17: "5",
    0x18: "=",
    0x19: "9",
    0x1A: "7",
    0x1B: "-",
    0x1C: "8",
    0x1D: "0",
    0x1E: "]",
    0x1F: "O",
    0x20: "U",
    0x21: "[",
    0x22: "I",
    0x23: "P",
    0x25: "L",
    0x26: "J",
    0x27: "'",
    0x28: "K",
    0x29: ";",
    0x2A: "\\",
    0x2B: ",",
    0x2C: "/",
    0x2D: "N",
    0x2E: "M",
    0x2F: ".",
    0x32: "`",
}


def _windows_code(key, injected: bool = False) -> int:
    """Number a key pressed on macOS or Linux as Windows would.

    Args:
        key: A pynput ``Key`` or ``KeyCode``.
        injected: Whether pynput marked the event injected.

    Returns:
        Its Windows virtual-key code, or UNKNOWN.
    """
    if isinstance(key, Key):
        return _OFF_WINDOWS.get(getattr(key, "name", None), UNKNOWN)
    if not isinstance(key, KeyCode):
        return UNKNOWN

    native = getattr(key, "vk", None)
    character = getattr(key, "char", None)
    darwin = _PLATFORM == "darwin"
    if darwin:
        keypad = _DARWIN_KEYPAD.get(native)
    else:
        if native is None and _BACKEND == "xorg" and not injected:
            native = _XORG_FOLDED_KEYPAD.get(character)
        keypad = _XORG_KEYPAD.get(native)
    if keypad is not None:
        return keypad

    if character in CHARACTERS:
        return CHARACTERS[character]

    if darwin:
        position = _DARWIN_KEYS.get(native)
        return CHARACTERS[position] if position else UNKNOWN
    # Not the native code on Linux: it is an X keysym under xorg but an
    # evdev code under uinput, and the two overlap. xorg reports the
    # character for every key that has one anyway.
    return UNKNOWN


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
    global keyboard, Key, KeyCode, _BACKEND
    if keyboard is None:
        from pynput import keyboard as module

        keyboard = module
        # "pynput.keyboard._xorg": the backend pynput chose, which on
        # Linux is xorg unless PYNPUT_BACKEND says uinput.
        _BACKEND = module.KeyCode.__module__.rpartition("._")[2]
    if Key is None:
        Key = keyboard.Key
    if KeyCode is None:
        KeyCode = keyboard.KeyCode


class Keyboard(EventSource):
    """Keyboard input for capturing key press and release events.

    Monitors key press and release events and converts them to numbers:
    a press emits the key's Windows virtual-key code, on every platform,
    and a release emits 0. So the arrow keys are 37 to 40 (left, up,
    right, down) on Windows, macOS and Linux alike. A key Windows has no
    code for emits -1.

    On macOS the app that runs Python -- Terminal, or your IDE -- needs
    Input Monitoring (System Settings > Privacy & Security). Without it
    no key arrives and nothing fails. pynput's warning that the process
    must be added to accessibility clients names the wrong permission.
    """

    class Configuration(EventSource.Configuration):
        """Configuration class for Keyboard source parameters."""

        class Keys(EventSource.Configuration.Keys):
            """Configuration key constants for the Keyboard source."""

            pass

    def __init__(self, edge_id: Optional[str] = None, **kwargs):
        """Initialize keyboard event source.

        Args:
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
            **kwargs: Additional configuration parameters for EventSource.
        """
        # pynput is imported in start(), not here and not at module
        # scope. A core has to be *constructible* in every residency --
        # a server builds one to describe the stream it is receiving,
        # and a document is deserialised wherever it is read -- while
        # only the process with the keyboard ever starts one. pynput
        # additionally raises at import on Linux with no display, so
        # importing at construction would make a headless process refuse
        # a document it was never going to read keys in.
        EventSource.__init__(self, edge_id=edge_id, **kwargs)

        # Initialize keyboard monitoring state
        self._running = False
        self._press_listener = None
        self._release_listener = None

    def _on_press(self, key, injected: bool = False):
        """Handle keyboard key press events.

        Extracts the Windows virtual-key code and triggers an event.

        pynput reports each platform's own numbering: the Up arrow is 38
        on Windows, 126 on macOS and the keysym 65362 under X, so a
        recording made on one meant nothing on another and a marker
        written for one never fired on the others. Windows' numbering is
        kept, translated to elsewhere, because it is what every example
        and every recording made so far already uses.

        Args:
            key: Pressed key object from pynput (KeyCode or Key).
            injected: Whether pynput marked the event injected. pynput
                passes it only to a callback that declares it, so this
                parameter is what gives the X keypad rule a way to tell
                a Controller's digit from the keypad.
        """
        if _PLATFORM != "win32":
            self.trigger(_windows_code(key, injected))
            return

        # Windows: pynput already reports virtual-key codes.
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
        # Where the library is actually needed, and the first point at
        # which this process has declared it owns a keyboard. See
        # __init__ for why it is not imported there.
        _load()

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
