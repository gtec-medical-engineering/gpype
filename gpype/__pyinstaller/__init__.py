# -*- coding: utf-8 -*-
"""PyInstaller hook entry point for gpype."""

import os


def get_hook_dirs():
    """Return the path to the directory containing the PyInstaller hooks."""
    return [os.path.dirname(__file__)]
