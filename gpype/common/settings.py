import os
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional
from xml.dom import minidom

#: File name of the settings document, in whichever directory holds it.
_FILE_NAME = "settings.xml"


def _fallback_settings_path() -> Optional[Path]:
    """Where settings go when the platform's own directory is unusable.

    Android is the case this exists for: there ``Path.home()`` can raise,
    and a home directory that resolves is not necessarily writable. The
    temp directory is the app's own on a mobile platform, and the one
    Python checks is writable.

    It is the directory the pipeline's log falls back to
    (``pipeline._fallback_log_directory``), but not for the same reason.
    The log falls back on a platform ioiocore does not know, whatever its
    home; settings fall back when their own directory cannot be named or
    created, on any platform -- Windows included, where that replaces a
    machine-wide directory with this user's temp directory.

    Returns:
        A path under the temp directory, or None -- settings held in
        memory only -- when no temp directory is usable either.
    """
    try:
        return Path(tempfile.gettempdir()) / "gtec" / "gPype" / _FILE_NAME
    except (OSError, RuntimeError):
        return None


def _fallback_warning(wanted: Optional[Path], used: Optional[Path]) -> str:
    """Say where the settings went instead, or that they will not last.

    Args:
        wanted: The file the platform's directory would have held, or None
            where that directory could not be named.
        used: The fallback file, or None where settings live in memory.

    Returns:
        The warning to give.
    """
    because = (
        f"{wanted.parent} could not be created"
        if wanted is not None
        else "no home directory could be named"
    )
    if used is not None:
        return (
            f"g.Pype settings are kept in {used}, because {because}. The "
            f"temp directory may be cleared; set GPYPE_SETTINGS_DIR to keep "
            f"them elsewhere."
        )
    return (
        f"g.Pype settings are kept in memory only, because {because} and "
        f"the temp directory is not writable either. Anything written to "
        f"gp.Settings, the OSCAR key included, is lost when this process "
        f"exits; set GPYPE_SETTINGS_DIR to keep it."
    )


class _Settings(dict):
    """Singleton settings manager for g.Pype application configuration.

    Manages persistent settings stored in XML format in platform-specific
    directories. Provides automatic type conversion, default values,
    and thread-safe singleton access.

    Storage locations:
    - Windows: %PROGRAMDATA%/gtec/gPype/settings.xml
    - macOS: ~/Library/Application Support/gtec/gPype/settings.xml
    - elsewhere: $XDG_CONFIG_HOME (or ~/.config)/gtec/gPype/settings.xml

    Where that directory cannot be determined or created -- Android, with
    no usable home -- the temp directory's gtec/gPype is used instead, and
    where that fails too the settings live in memory only (``file_path``
    is None). Either is announced once, with a RuntimeWarning naming the
    file used or saying the settings will not persist. An explicit
    ``GPYPE_SETTINGS_DIR`` is never replaced.

    Use Settings.get() to access the singleton instance.
    """

    #: Default settings applied on first initialization
    DEFAULTS = {"Key": ""}

    #: Singleton instance reference
    _instance: Optional["_Settings"] = None

    def __init__(self):
        """Initialize the settings singleton instance.

        Loads existing settings from XML file, applies default values
        for missing keys, and saves the updated configuration.

        Raises:
            RuntimeError: If an instance already exists. Use get() instead.
        """
        # Enforce singleton pattern
        if _Settings._instance is not None:
            raise RuntimeError(
                "Use Settings.get_instance() to access the "
                "singleton instance."
            )

        # Initialize dictionary base class
        super().__init__()

        # Set up file path and ensure directory exists
        self.file_path = self._get_settings_path()
        self._ensure_path_exists()

        # Load existing settings from file
        loaded_settings = self._read()
        self.update(loaded_settings)

        # Apply default values for missing keys
        updated = False
        for key, value in _Settings.DEFAULTS.items():
            if key not in self:
                self[key] = value
                updated = True

        # Save updated settings if defaults were added
        if updated:
            self.write()

        # Register as singleton instance
        _Settings._instance = self

    @staticmethod
    def get() -> "_Settings":
        """Get the singleton settings instance.

        Creates the instance if it doesn't exist, otherwise returns
        the existing instance.

        Returns:
            _Settings: The singleton settings instance.
        """
        if _Settings._instance is None:
            _Settings()
        return _Settings._instance

    def _get_settings_path(self) -> Optional[Path]:
        """Determine platform-specific settings file path.

        Returns:
            Path: Full path to the settings XML file, or None when the
            platform's directory cannot be determined -- no home
            directory, as on Android. ``_ensure_path_exists`` then falls
            back.
        """
        # Check for environment variable override (useful for testing)
        if "GPYPE_SETTINGS_DIR" in os.environ:
            base = Path(os.environ["GPYPE_SETTINGS_DIR"])
            return base / _FILE_NAME

        try:
            if sys.platform == "win32":
                # Windows: Use PROGRAMDATA for system-wide settings
                base = Path(os.getenv("PROGRAMDATA", r"C:\ProgramData"))
            elif sys.platform == "darwin":
                # macOS: Use user's Application Support directory
                base = Path.home() / "Library" / "Application Support"
            else:
                # Everything else follows the XDG convention. Settings is
                # constructed at import, so raising here would make the
                # whole package unimportable rather than merely
                # unsupported, and ioiocore itself runs on Linux.
                xdg = os.getenv("XDG_CONFIG_HOME")
                base = Path(xdg) if xdg else Path.home() / ".config"
        except (KeyError, OSError, RuntimeError):
            # Path.home() raises where there is no home to name: no HOME
            # and no password-database entry, which is Android's normal
            # state (RuntimeError from 3.12; KeyError before).
            return None

        return base / "gtec" / "gPype" / _FILE_NAME

    def _ensure_path_exists(self):
        """Create the settings directory, falling back where it cannot be.

        The platform's own directory first. Where it could not be named,
        or cannot be created, the temp directory's; where that cannot be
        created either, ``file_path`` becomes None and the settings live
        in memory for this process. An explicit ``GPYPE_SETTINGS_DIR`` is
        created or refused as given: a directory someone chose is not
        silently replaced by another.

        A fallback warns, once, since this runs once per process: it
        moves the settings somewhere nobody chose, and in memory a key
        written through ``gp.Settings`` is gone at exit. Without the
        warning both happened silently.
        """
        if self.file_path is not None:
            try:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                return
            except OSError:
                if "GPYPE_SETTINGS_DIR" in os.environ:
                    raise

        fallback = _fallback_settings_path()
        if fallback is not None:
            try:
                fallback.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                fallback = None
        warnings.warn(
            _fallback_warning(self.file_path, fallback),
            RuntimeWarning,
            stacklevel=2,
        )
        self.file_path = fallback

    def _convert_type(self, value: str) -> Any:
        """Convert string values to appropriate Python types.

        Args:
            value (str): String value to convert.

        Returns:
            Any: Converted value (bool, int, float, or str).
        """
        val = value.strip().lower()

        # Convert boolean values
        if val == "true":
            return True
        if val == "false":
            return False

        # Try integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Default to string
        return value

    def _read(self) -> dict[str, Any]:
        """Read settings from the XML file.

        Returns:
            dict[str, Any]: Dictionary of setting key-value pairs.
        """
        # Return empty dict if file doesn't exist, or there is none
        if self.file_path is None or not self.file_path.exists():
            return {}

        try:
            # Parse XML file
            tree = ET.parse(self.file_path)
            root = tree.getroot()

            # Convert each XML element to key-value pair with type conversion
            return {
                child.tag: self._convert_type(child.text or "")
                for child in root
            }
        except Exception as e:
            # Log warning but continue with empty settings
            print(f"Warning: Failed to parse settings file: {e}")
            return {}

    def write(self):
        """Write current settings to the XML file with pretty formatting.

        Does nothing where no directory could be created (``file_path``
        is None): the settings are then kept for this process only, as
        the warning given at construction said.
        """
        if self.file_path is None:
            return

        # Create XML root element
        root = ET.Element("Settings")

        # Add each setting as a child element
        for key, value in self.items():
            elem = ET.SubElement(root, key)
            elem.text = str(value)

        # Convert to pretty-formatted XML string
        rough_string = ET.tostring(root, "utf-8")
        pretty_string = minidom.parseString(rough_string).toprettyxml(
            indent="  "
        )

        # Write to file with UTF-8 encoding
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(pretty_string)


#: Global settings instance for convenient access throughout the application
Settings: _Settings = _Settings()
