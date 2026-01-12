# -*- coding: utf-8 -*-
"""PyInstaller hook for gpype.

This hook ensures that all required dependencies are properly included
when building a standalone executable with PyInstaller.
"""

hiddenimports = [
    'pyqtgraph',
    'PySide6',
    'scipy',
    'ioiocore',
    'gtec_licensing',
]
