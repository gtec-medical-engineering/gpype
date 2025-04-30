import sys
if sys.platform != "win32":
    raise NotImplementedError("This module is only supported on Windows.")

from PySide6.QtWidgets import (
    QWidget, QPushButton, QComboBox, QFileDialog,
    QLabel, QSizePolicy, QSpacerItem, QMessageBox,
    QHBoxLayout
)

from .base.widget import Widget
import os
import glob

MINIMUM_BUTTON_WIDTH = 120


class ParadigmPresenter(Widget):

    def __init__(self,
                 root_folder: str = None):
        import gtec_pp as pp
        super().__init__(widget=QWidget(), name="ParadigmPresenter Control",
                         layout=QHBoxLayout)

        self.paradigm_presenter = pp.ParadigmPresenter()

        self._root_folder = root_folder

        self.start_button = QPushButton("Start Paradigm")
        self.start_button.setMinimumWidth(MINIMUM_BUTTON_WIDTH)
        self.start_button.clicked.connect(self._start_paradigm)

        self.load_button = None
        self.dropdown = None
        if root_folder is None:
            self.load_button = QPushButton("Load Paradigm...")
            self.load_button.setMinimumWidth(MINIMUM_BUTTON_WIDTH)
            self.load_button.clicked.connect(self._load_paradigm)
            self.start_button.setEnabled(False)
            self._layout.addWidget(self.load_button)
        else:
            label = QLabel("Select Paradigm:")
            self.dropdown = QComboBox()

            self.paradigms = self._get_all_paradigms()
            if len(self.paradigms) > 0:
                self.dropdown.addItems(self.paradigms)
            else:
                QMessageBox.critical(
                    QWidget(),  # or `self` if inside a QWidget subclass
                    "No Paradigms found",
                    f"No Paradigms found in: {self._root_folder}"
                )
                self.start_button.setEnabled(False)

            self.dropdown.setMinimumWidth(2 * MINIMUM_BUTTON_WIDTH)
            self.dropdown.currentIndexChanged.connect(self._select_paradigm)
            if len(self.paradigms) > 0:
                paradigm_file = os.path.join(self._root_folder,
                                             self.paradigms[0])
                self.paradigm_presenter.load_paradigm(paradigm_file)
            self._layout.addWidget(label)
            self._layout.addWidget(self.dropdown)

        self._layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Paradigm")
        self.stop_button.setMinimumWidth(MINIMUM_BUTTON_WIDTH)
        self.stop_button.clicked.connect(self._stop_paradigm)
        self.stop_button.setEnabled(False)
        self._layout.addWidget(self.stop_button)
        self._layout.addSpacerItem(QSpacerItem(0,
                                               0,
                                               QSizePolicy.Expanding,
                                               QSizePolicy.Minimum))

        type_plain = self.paradigm_presenter.constants.WINDOWTYPE_PLAIN
        self.paradigm_presenter.open_window(type_plain)

    def _start_paradigm(self):
        if self.dropdown:
            self.dropdown.setEnabled(False)
        self.start_button.setEnabled(False)
        self.paradigm_presenter.start_paradigm()
        self.stop_button.setEnabled(True)

    def _stop_paradigm(self):
        self.stop_button.setEnabled(False)
        self.paradigm_presenter.stop_paradigm()
        self.start_button.setEnabled(True)
        if self.dropdown:
            self.dropdown.setEnabled(True)

    def _select_paradigm(self):
        idx = self.dropdown.currentIndex()
        paradigm_file = os.path.join(self._root_folder, self.paradigms[idx])
        if self.paradigm_presenter.load_paradigm(paradigm_file):
            self.start_button.setEnabled(True)

    def _load_paradigm(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Paradigm File (*.xml)")

        # Show the dialog and get the selected file paths
        self.paradigms = None
        if file_dialog.exec():
            paradigm_file = file_dialog.selectedFiles()
            if paradigm_file is not None and len(paradigm_file) > 0:
                if self.paradigm_presenter.load_paradigm(paradigm_file[0]):
                    self.start_button.setEnabled(True)

    def _get_all_paradigms(self):
        paradigms = []

        # Recursively get all directories
        dirs = [root for root, _, _ in os.walk(self._root_folder)]
        if not dirs:
            return paradigms

        for dir_path in dirs:
            xml_files = glob.glob(os.path.join(dir_path, "*.xml"))
            subdir = os.path.relpath(dir_path, self._root_folder)
            subdir = subdir + os.sep if subdir != "." else ""

            if xml_files:
                paradigms.extend([os.path.join(subdir, os.path.basename(f))
                                  for f in xml_files])

        return self._validate_paradigms(paradigms)

    def _validate_paradigms(self, paradigms):
        valid_paradigms = []

        for paradigm in paradigms:
            cur_paradigm = os.path.join(self._root_folder, paradigm)
            try:
                self.paradigm_presenter.load_paradigm(cur_paradigm)
                valid_paradigms.append(paradigm)
            except Exception:
                pass  # Ignore invalid paradigms

        return valid_paradigms

    def terminate(self):
        self.paradigm_presenter.close_windows()
        self.paradigm_presenter.shutdown()
        super().terminate()
