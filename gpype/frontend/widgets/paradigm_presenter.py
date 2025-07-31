# Standard library imports
import glob
import os
import sys

# Platform check - this module is Windows-only
if sys.platform != "win32":
    raise NotImplementedError("This module is only supported on Windows.")

# Third-party imports
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QSizePolicy,
                               QSpacerItem, QWidget)

# Local imports
from .base.widget import Widget

# UI constants
MINIMUM_BUTTON_WIDTH = 120


class ParadigmPresenter(Widget):
    """
    Control widget for g.tec ParadigmPresenter integration.

    Provides a user interface for loading, starting, and stopping paradigms
    using the g.tec ParadigmPresenter library. Supports both folder-based
    paradigm selection and manual file loading.
    """

    def __init__(self, paradigm: str = None):
        """
        Initialize the ParadigmPresenter control widget.

        Args:
            paradigm (str, optional): Path to paradigm file (.xml) or folder
                containing paradigm files. If file, it will be loaded directly.
                If folder, a dropdown will be created for paradigm selection.
                If None, a file dialog will be used for loading.
        """
        # Import gtec_pp here to avoid issues if not installed
        import gtec_pp as pp

        # Initialize base widget with horizontal layout
        super().__init__(
            widget=QWidget(),
            name="ParadigmPresenter Control",
            layout=QHBoxLayout,
        )

        # Initialize the ParadigmPresenter instance
        self.paradigm_presenter = pp.ParadigmPresenter()

        # Determine if paradigm is a file or folder
        self._paradigm_path = paradigm
        self._root_folder = None
        self._paradigm_file = None

        if paradigm:
            if os.path.isfile(paradigm) and paradigm.endswith(".xml"):
                # Single paradigm file provided
                self._paradigm_file = paradigm
            elif os.path.isdir(paradigm):
                # Folder with paradigms provided
                self._root_folder = paradigm

        # Initialize UI components
        self._setup_ui()

        # Open the paradigm presenter window
        self._open_presenter_window()

    def _setup_ui(self):
        """Set up the user interface components."""
        # Create main control buttons
        self._create_start_button()
        self._create_paradigm_selection()
        self._create_stop_button()

        # Add flexible spacer to push buttons to the left
        self._layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

    def _create_start_button(self):
        """Create and configure the start paradigm button."""
        self.start_button = QPushButton("Start Paradigm")
        self.start_button.setMinimumWidth(MINIMUM_BUTTON_WIDTH)
        self.start_button.clicked.connect(self._start_paradigm)

    def _create_paradigm_selection(self):
        """Create paradigm selection UI (dropdown, file display, or load)."""
        self.load_button = None
        self.dropdown = None

        if self._paradigm_file:
            # Single paradigm file provided - display filename and load it
            self._create_paradigm_file_display()
        elif self._root_folder:
            # Root folder provided - create dropdown for selection
            self._create_paradigm_dropdown()
        else:
            # No paradigm specified - use file dialog for loading
            self._create_load_button()

    def _create_load_button(self):
        """Create the load paradigm button for file dialog selection."""
        self.load_button = QPushButton("Load Paradigm...")
        self.load_button.setMinimumWidth(MINIMUM_BUTTON_WIDTH)
        self.load_button.clicked.connect(self._load_paradigm)
        self.start_button.setEnabled(False)  # Disabled until paradigm loaded
        self._layout.addWidget(self.load_button)

    def _create_paradigm_file_display(self):
        """Create display for single paradigm file and load it."""
        # Display the paradigm filename
        label = QLabel("Paradigm:")
        filename_label = QLabel(os.path.basename(self._paradigm_file))
        filename_label.setMinimumWidth(2 * MINIMUM_BUTTON_WIDTH)

        # Try to load the paradigm file
        try:
            if self.paradigm_presenter.load_paradigm(self._paradigm_file):
                self.start_button.setEnabled(True)
            else:
                # Failed to load - show error and disable start
                QMessageBox.critical(
                    QWidget(),
                    "Failed to Load Paradigm",
                    f"Could not load paradigm file: {self._paradigm_file}",
                )
                self.start_button.setEnabled(False)
        except Exception as e:
            # Exception during loading - show error and disable start
            QMessageBox.critical(
                QWidget(),
                "Paradigm Load Error",
                f"Error loading paradigm: {str(e)}",
            )
            self.start_button.setEnabled(False)

        # Add components to layout
        self._layout.addWidget(label)
        self._layout.addWidget(filename_label)

    def _create_paradigm_dropdown(self):
        """Create dropdown for paradigm selection from root folder."""
        label = QLabel("Select Paradigm:")
        self.dropdown = QComboBox()

        # Discover available paradigms in the root folder
        self.paradigms = self._get_all_paradigms()

        if len(self.paradigms) > 0:
            # Configure dropdown with found paradigms
            self.dropdown.addItems(self.paradigms)
            self.dropdown.setMinimumWidth(2 * MINIMUM_BUTTON_WIDTH)
            self.dropdown.currentIndexChanged.connect(self._select_paradigm)

            # Load the first paradigm by default
            paradigm_file = os.path.join(self._root_folder, self.paradigms[0])
            self.paradigm_presenter.load_paradigm(paradigm_file)

            # Add components to layout
            self._layout.addWidget(label)
            self._layout.addWidget(self.dropdown)
        else:
            # No paradigms found - show error and disable start button
            QMessageBox.critical(
                QWidget(),
                "No Paradigms found",
                f"No Paradigms found in: {self._root_folder}",
            )
            self.start_button.setEnabled(False)
            self._layout.addWidget(label)  # Add label even without dropdown

    def _create_stop_button(self):
        """Create and configure the stop paradigm button."""
        # Add start button to layout (positioned after paradigm selection)
        self._layout.addWidget(self.start_button)

        # Create stop button
        self.stop_button = QPushButton("Stop Paradigm")
        self.stop_button.setMinimumWidth(MINIMUM_BUTTON_WIDTH)
        self.stop_button.clicked.connect(self._stop_paradigm)
        self.stop_button.setEnabled(False)  # Disabled until paradigm starts
        self._layout.addWidget(self.stop_button)

    def _open_presenter_window(self):
        """Open the ParadigmPresenter window."""
        window_type = self.paradigm_presenter.constants.WINDOWTYPE_PLAIN
        self.paradigm_presenter.open_window(window_type)

    def _start_paradigm(self):
        """Start the currently loaded paradigm and update UI state."""
        # Disable controls during paradigm execution
        if self.dropdown:
            self.dropdown.setEnabled(False)
        if self.load_button:
            self.load_button.setEnabled(False)
        self.start_button.setEnabled(False)

        # Start the paradigm
        self.paradigm_presenter.start_paradigm()

        # Enable stop button
        self.stop_button.setEnabled(True)

    def _stop_paradigm(self):
        """Stop the currently running paradigm and update UI state."""
        # Disable stop button immediately
        self.stop_button.setEnabled(False)

        # Stop the paradigm
        self.paradigm_presenter.stop_paradigm()

        # Re-enable controls
        self.start_button.setEnabled(True)
        if self.dropdown:
            self.dropdown.setEnabled(True)
        if self.load_button:
            self.load_button.setEnabled(True)

    def _select_paradigm(self):
        """Handle paradigm selection from dropdown."""
        # Get selected paradigm index and construct file path
        idx = self.dropdown.currentIndex()
        paradigm_file = os.path.join(self._root_folder, self.paradigms[idx])

        # Load the selected paradigm and enable start button if successful
        if self.paradigm_presenter.load_paradigm(paradigm_file):
            self.start_button.setEnabled(True)

    def _load_paradigm(self):
        """Open file dialog to load a paradigm file."""
        # Configure file dialog for XML paradigm files
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Paradigm File (*.xml)")

        # Show dialog and handle file selection
        if file_dialog.exec():
            paradigm_files = file_dialog.selectedFiles()
            if paradigm_files and len(paradigm_files) > 0:
                # Load the first selected paradigm file
                if self.paradigm_presenter.load_paradigm(paradigm_files[0]):
                    self.start_button.setEnabled(True)

    def _get_all_paradigms(self):
        """
        Discover all paradigm files in the root folder and subfolders.

        Returns:
            list: List of relative paths to valid paradigm XML files.
        """
        paradigms = []

        # Recursively walk through all directories in root folder
        dirs = [root for root, _, _ in os.walk(self._root_folder)]
        if not dirs:
            return paradigms

        # Search for XML files in each directory
        for dir_path in dirs:
            xml_files = glob.glob(os.path.join(dir_path, "*.xml"))

            # Calculate relative path from root folder
            subdir = os.path.relpath(dir_path, self._root_folder)
            subdir = subdir + os.sep if subdir != "." else ""

            # Add found XML files with relative paths
            if xml_files:
                paradigms.extend(
                    [
                        os.path.join(subdir, os.path.basename(f))
                        for f in xml_files
                    ]
                )

        # Validate paradigms before returning
        return self._validate_paradigms(paradigms)

    def _validate_paradigms(self, paradigms):
        """
        Validate paradigm files by attempting to load them.

        Args:
            paradigms (list): List of paradigm file paths to validate.

        Returns:
            list: List of valid paradigm file paths.
        """
        valid_paradigms = []

        for paradigm in paradigms:
            paradigm_path = os.path.join(self._root_folder, paradigm)
            try:
                # Attempt to load paradigm to validate it
                self.paradigm_presenter.load_paradigm(paradigm_path)
                valid_paradigms.append(paradigm)
            except Exception:
                # Silently ignore invalid paradigms
                pass

        return valid_paradigms

    def terminate(self):
        """
        Clean up resources when the widget is terminated.

        Closes ParadigmPresenter windows and shuts down the presenter
        before calling the parent terminate method.
        """
        # Clean up ParadigmPresenter resources
        self.paradigm_presenter.close_windows()
        self.paradigm_presenter.shutdown()

        # Call parent cleanup
        super().terminate()
