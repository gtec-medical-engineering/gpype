import os
from typing import TYPE_CHECKING

import ioiocore as ioc

from .__version__ import __version__

_LAZY_IMPORTS = {}

# ----------------------------------------------
# top level & common

if TYPE_CHECKING:
    from .backend.pipeline import Pipeline
    from .common.constants import Constants
    from .common.settings import Settings
    from .frontend.main_app import MainApp

_LAZY_IMPORTS.update({
    "backend.pipeline": "Pipeline",
    "frontend.main_app": "MainApp",
    "common.constants": "Constants",
    "common.settings": "Settings"
})

# ----------------------------------------------
# backend.core

if TYPE_CHECKING:
    from .backend.core.i_node import INode
    from .backend.core.i_port import IPort
    from .backend.core.io_node import IONode
    from .backend.core.node import Node
    from .backend.core.o_node import ONode
    from .backend.core.o_port import OPort

_LAZY_IMPORTS.update({
    "backend.core.i_node": "INode",
    "backend.core.i_port": "IPort",
    "backend.core.io_node": "IONode",
    "backend.core.node": "Node",
    "backend.core.o_node": "ONode",
    "backend.core.o_port": "OPort",
})

# ----------------------------------------------
# backend.filters

if TYPE_CHECKING:
    from .backend.filters.bandpass import Bandpass
    from .backend.filters.bandstop import Bandstop
    from .backend.filters.base.generic_filter import GenericFilter
    from .backend.filters.highpass import Highpass
    from .backend.filters.lowpass import Lowpass
    from .backend.filters.moving_average import MovingAverage

_LAZY_IMPORTS.update({
    "backend.filters.bandpass": "Bandpass",
    "backend.filters.bandstop": "Bandstop",
    "backend.filters.base.generic_filter": "GenericFilter",
    "backend.filters.highpass": "Highpass",
    "backend.filters.lowpass": "Lowpass",
    "backend.filters.moving_average": "MovingAverage",
})

# ----------------------------------------------
# backend.flow

if TYPE_CHECKING:
    from .backend.flow.framer import Framer
    from .backend.flow.router import Router
    from .backend.flow.trigger import Trigger

_LAZY_IMPORTS.update({
    "backend.flow.framer": "Framer",
    "backend.flow.router": "Router",
    "backend.flow.trigger": "Trigger",
})

# ----------------------------------------------
# backend.sinks

if TYPE_CHECKING:
    from .backend.sinks.file_writer import FileWriter
    from .backend.sinks.lsl_sender import LSLSender
    from .backend.sinks.udp_sender import UDPSender

_LAZY_IMPORTS.update({
    "backend.sinks.file_writer": "FileWriter",
    "backend.sinks.lsl_sender": "LSLSender",
    "backend.sinks.udp_sender": "UDPSender",
})

# ----------------------------------------------
# backend.sources

if TYPE_CHECKING:
    from .backend.sources.bci_core8 import BCICore8
    from .backend.sources.g_nautilus import GNautilus
    from .backend.sources.generator import Generator
    from .backend.sources.keyboard import Keyboard
    from .backend.sources.udp_receiver import UDPReceiver

_LAZY_IMPORTS.update({
    "backend.sources.bci_core8": "BCICore8",
    "backend.sources.generator": "Generator",
    "backend.sources.keyboard": "Keyboard",
    "backend.sources.udp_receiver": "UDPReceiver",
    "backend.sources.g_nautilus": "GNautilus",
})

# ----------------------------------------------
# backend.timing

if TYPE_CHECKING:
    from .backend.timing.decimator import Decimator
    from .backend.timing.delay import Delay
    from .backend.timing.hold import Hold

_LAZY_IMPORTS.update({
    "backend.timing.decimator": "Decimator",
    "backend.timing.delay": "Delay",
    "backend.timing.hold": "Hold",
})

# ----------------------------------------------
# backend.transform

if TYPE_CHECKING:
    from .backend.transform.equation import Equation
    from .backend.transform.fft import FFT

_LAZY_IMPORTS.update({
    "backend.transform.equation": "Equation",
    "backend.transform.fft": "FFT",
})

# ----------------------------------------------
# frontend.widgets

if TYPE_CHECKING:
    from .frontend.widgets.paradigm_presenter import ParadigmPresenter
    from .frontend.widgets.performance_monitor import PerformanceMonitor
    from .frontend.widgets.spectrum_scope import SpectrumScope
    from .frontend.widgets.time_series_scope import TimeSeriesScope
    from .frontend.widgets.trigger_scope import TriggerScope

_LAZY_IMPORTS.update({
    "frontend.widgets.paradigm_presenter": "ParadigmPresenter",
    "frontend.widgets.performance_monitor": "PerformanceMonitor",
    "frontend.widgets.spectrum_scope": "SpectrumScope",
    "frontend.widgets.time_series_scope": "TimeSeriesScope",
    "frontend.widgets.trigger_scope": "TriggerScope",
})

# ==============================================


def __getattr__(name):
    """Lazy import handler - imports modules only when accessed."""
    # Find the module path for the requested name
    module_path = None
    for path, class_name in _LAZY_IMPORTS.items():
        if class_name == name:
            module_path = path
            break

    if module_path:
        try:
            # Import the module and get the class/function
            module = __import__(f"gpype.{module_path}", fromlist=[name])
            attr = getattr(module, name)

            # Cache it in globals for faster subsequent access
            globals()[name] = attr
            return attr
        except (ImportError, AttributeError) as e:
            raise AttributeError(f"Cannot import '{name}' from gpype: {e}")

    raise AttributeError(f"module 'gpype' has no attribute '{name}'")


def __dir__():
    """Return all available attributes for autocomplete."""
    return list(globals().keys()) + list(_LAZY_IMPORTS.values())


# Add gpype as preinstalled module to ioiocore
ioc.Portable.add_preinstalled_module("gpype")
