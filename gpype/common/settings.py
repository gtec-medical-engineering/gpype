import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional
from xml.dom import minidom


class _Settings(dict):
    """Singleton settings manager for g.Pype application configuration.

    Manages persistent settings stored in XML format in platform-specific
    directories. Provides automatic type conversion, default values,
    and thread-safe singleton access.

    Storage locations:
    - Windows: %PROGRAMDATA%/gtec/gPype/settings.xml
    - macOS: ~/Library/Application Support/gtec/gPype/settings.xml

    Note:
        Use Settings.get() to access the singleton instance.
    """

    # Default settings applied on first initialization
    DEFAULTS = {"Key": ""}

    # Singleton instance reference
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

    def _get_settings_path(self) -> Path:
        """Determine platform-specific settings file path.

        Returns:
            Path: Full path to the settings XML file.

        Raises:
            RuntimeError: If the operating system is not supported.
        """
        # Check for environment variable override (useful for testing)
        if "GPYPE_SETTINGS_DIR" in os.environ:
            base = Path(os.environ["GPYPE_SETTINGS_DIR"])
            return base / "settings.xml"

        if sys.platform == "win32":
            # Windows: Use PROGRAMDATA for system-wide settings
            base = Path(os.getenv("PROGRAMDATA", r"C:\ProgramData"))
        elif sys.platform == "darwin":
            # macOS: Use user's Application Support directory
            base = Path.home() / "Library" / "Application Support"
        else:
            # Unsupported platform
            raise RuntimeError("Unsupported OS")

        return base / "gtec" / "gPype" / "settings.xml"

    def _ensure_path_exists(self):
        """Create the settings directory if it doesn't exist."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

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
        # Return empty dict if file doesn't exist
        if not self.file_path.exists():
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
        """Write current settings to the XML file with pretty formatting."""
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


# Global settings instance for convenient access throughout the application
# Use Settings.get() for type-safe access or Settings directly for quick access
Settings: _Settings = _Settings()
