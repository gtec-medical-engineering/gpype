import os
from typing import TYPE_CHECKING

import ioiocore as ioc

from .__version__ import __version__

#: Version of the public control surface, independent of the release
#: version. A tool driving g.Pype from outside -- a control plane, an
#: authoring studio -- pins against this rather than against __version__,
#: which changes on every release whether the surface moved or not.
#: Bump the minor part when something is added, the major part when
#: something already published changes shape or disappears.
API_VERSION = "1.0"

_LAZY_IMPORTS = {}

# ----------------------------------------------
# top level & common

if TYPE_CHECKING:  # pragma: no cover
    from .backend.pipeline import Pipeline
    from .common._private.allowlist import allow_modules
    from .common.constants import Constants
    from .common.document import canonicalize
    from .common.launch_config import LaunchConfig
    from .common.montage import Montage
    from .common.result import Result
    from .common.settings import Settings
    from .frontend.main_app import MainApp

_LAZY_IMPORTS.update(
    {
        "backend.pipeline": "Pipeline",
        "frontend.main_app": "MainApp",
        "common._private.allowlist": "allow_modules",
        "common.document": "canonicalize",
        "common.constants": "Constants",
        "common.launch_config": "LaunchConfig",
        "common.montage": "Montage",
        "common.result": "Result",
        "common.settings": "Settings",
    }
)

# ----------------------------------------------
# backend.core

if TYPE_CHECKING:  # pragma: no cover
    from .backend.core._private.controllable import action, controllable
    from .backend.core.i_node import INode
    from .backend.core.i_port import IPort
    from .backend.core.io_node import IONode
    from .backend.core.node import Node
    from .backend.core.o_node import ONode
    from .backend.core.o_port import OPort

_LAZY_IMPORTS.update(
    {
        # One module, two symbols -- and _LAZY_IMPORTS is keyed by module
        # path while __getattr__ reverse-looks-up by symbol name, so a
        # module can carry only one mapped name. backend.core re-exports
        # both decorators, so `action` is routed through the package
        # rather than teaching the map a second value shape: test_init
        # asserts every value is a str.
        "backend.core._private.controllable": "controllable",
        "backend.core": "action",
        "backend.core.i_node": "INode",
        "backend.core.i_port": "IPort",
        "backend.core.io_node": "IONode",
        "backend.core.node": "Node",
        "backend.core.o_node": "ONode",
        "backend.core.o_port": "OPort",
    }
)

# ----------------------------------------------
# chain layer, re-exported from ioiocore
#
# gpype's own sources, sinks and widgets are ioiocore chains, so the
# containers are part of gpype's public API. These are ioiocore types
# rather than gpype modules, so they cannot go through _LAZY_IMPORTS;
# bind them directly.

Chain = ioc.Chain
IChain = ioc.IChain
IOChain = ioc.IOChain
OChain = ioc.OChain

# ----------------------------------------------
# backend.filters

if TYPE_CHECKING:  # pragma: no cover
    from .backend.filters.bandpass import Bandpass
    from .backend.filters.bandstop import Bandstop
    from .backend.filters.base.generic_filter import GenericFilter
    from .backend.filters.highpass import Highpass
    from .backend.filters.lowpass import Lowpass
    from .backend.filters.moving_average import MovingAverage

_LAZY_IMPORTS.update(
    {
        "backend.filters.bandpass": "Bandpass",
        "backend.filters.bandstop": "Bandstop",
        "backend.filters.base.generic_filter": "GenericFilter",
        "backend.filters.highpass": "Highpass",
        "backend.filters.lowpass": "Lowpass",
        "backend.filters.moving_average": "MovingAverage",
    }
)

# ----------------------------------------------
# backend.flow

if TYPE_CHECKING:  # pragma: no cover
    from .backend.flow.channel_labeler import ChannelLabeler
    from .backend.flow.channel_selector import ChannelSelector
    from .backend.flow.epoch_average import EpochAverage
    from .backend.flow.framer import Framer
    from .backend.flow.router import Router
    from .backend.flow.threshold import Threshold
    from .backend.flow.trigger import Trigger

_LAZY_IMPORTS.update(
    {
        "backend.flow.channel_labeler": "ChannelLabeler",
        "backend.flow.channel_selector": "ChannelSelector",
        "backend.flow.epoch_average": "EpochAverage",
        "backend.flow.framer": "Framer",
        "backend.flow.router": "Router",
        "backend.flow.threshold": "Threshold",
        "backend.flow.trigger": "Trigger",
    }
)

# ----------------------------------------------
# backend.sinks

if TYPE_CHECKING:  # pragma: no cover
    from .backend.sinks.collector import Collector
    from .backend.sinks.csv_writer import CsvWriter
    from .backend.sinks.edf_writer import EDFWriter
    from .backend.sinks.hdf5_writer import HDF5Writer
    from .backend.sinks.lsl_sender import LSLSender
    from .backend.sinks.mat_writer import MatWriter
    from .backend.sinks.udp_sender import UDPSender

_LAZY_IMPORTS.update(
    {
        "backend.sinks.collector": "Collector",
        "backend.sinks.csv_writer": "CsvWriter",
        "backend.sinks.edf_writer": "EDFWriter",
        "backend.sinks.hdf5_writer": "HDF5Writer",
        "backend.sinks.lsl_sender": "LSLSender",
        "backend.sinks.mat_writer": "MatWriter",
        "backend.sinks.udp_sender": "UDPSender",
    }
)

# ----------------------------------------------
# backend.sources

if TYPE_CHECKING:  # pragma: no cover
    from .backend.sources.bci_core import BCICore
    from .backend.sources.bci_core8 import BCICore8
    from .backend.sources.csv_reader import CsvReader
    from .backend.sources.edf_reader import EDFReader
    from .backend.sources.g_hiamp import GHIamp
    from .backend.sources.g_nautilus import GNautilus
    from .backend.sources.g_usbamp import GUSBamp
    from .backend.sources.generator import Generator
    from .backend.sources.gtc_reader import GtcReader
    from .backend.sources.hdf5_reader import HDF5Reader
    from .backend.sources.hybrid_black import HybridBlack
    from .backend.sources.keyboard import Keyboard
    from .backend.sources.lsl_receiver import LslReceiver
    from .backend.sources.marker import Marker
    from .backend.sources.mat_reader import MatReader
    from .backend.sources.udp_receiver import UDPReceiver

_LAZY_IMPORTS.update(
    {
        "backend.sources.bci_core": "BCICore",
        # The old name, kept resolvable. It needs its own module rather
        # than an alias entry because this map carries one symbol per
        # module and the serialization allow-list is derived from its
        # keys -- see bci_core8.py.
        "backend.sources.bci_core8": "BCICore8",
        "backend.sources.csv_reader": "CsvReader",
        "backend.sources.edf_reader": "EDFReader",
        "backend.sources.g_hiamp": "GHIamp",
        "backend.sources.hdf5_reader": "HDF5Reader",
        "backend.sources.mat_reader": "MatReader",
        "backend.sources.g_nautilus": "GNautilus",
        "backend.sources.g_usbamp": "GUSBamp",
        "backend.sources.generator": "Generator",
        "backend.sources.gtc_reader": "GtcReader",
        "backend.sources.hybrid_black": "HybridBlack",
        "backend.sources.keyboard": "Keyboard",
        "backend.sources.lsl_receiver": "LslReceiver",
        "backend.sources.marker": "Marker",
        "backend.sources.udp_receiver": "UDPReceiver",
    }
)

# ----------------------------------------------
# backend.timing

if TYPE_CHECKING:  # pragma: no cover
    from .backend.timing.decimator import Decimator
    from .backend.timing.delay import Delay
    from .backend.timing.interpolator import Interpolator

_LAZY_IMPORTS.update(
    {
        "backend.timing.decimator": "Decimator",
        "backend.timing.delay": "Delay",
        "backend.timing.interpolator": "Interpolator",
    }
)

# ----------------------------------------------
# backend.transform

if TYPE_CHECKING:  # pragma: no cover
    from .backend.transform import node
    from .backend.transform.apply import Apply
    from .backend.transform.band_power import BandPower
    from .backend.transform.baseline import Baseline
    from .backend.transform.equation import Equation
    from .backend.transform.fft import FFT
    from .backend.transform.reference import Reference
    from .backend.transform.rolling_statistic import RollingStatistic

_LAZY_IMPORTS.update(
    {
        "backend.transform": "node",
        "backend.transform.apply": "Apply",
        "backend.transform.band_power": "BandPower",
        "backend.transform.baseline": "Baseline",
        "backend.transform.equation": "Equation",
        "backend.transform.fft": "FFT",
        "backend.transform.reference": "Reference",
        "backend.transform.rolling_statistic": "RollingStatistic",
    }
)

# ----------------------------------------------
# frontend.widgets

if TYPE_CHECKING:  # pragma: no cover
    from .frontend.widgets.paradigm_presenter import ParadigmPresenter
    from .frontend.widgets.result_scope import ResultScope
    from .frontend.widgets.spectrum_scope import SpectrumScope
    from .frontend.widgets.time_series_scope import TimeSeriesScope
    from .frontend.widgets.trigger_scope import TriggerScope

_LAZY_IMPORTS.update(
    {
        "frontend.widgets.paradigm_presenter": "ParadigmPresenter",
        "frontend.widgets.result_scope": "ResultScope",
        "frontend.widgets.spectrum_scope": "SpectrumScope",
        "frontend.widgets.time_series_scope": "TimeSeriesScope",
        "frontend.widgets.trigger_scope": "TriggerScope",
    }
)

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
            hint = _missing_extra_hint(e)
            if hint:
                raise AttributeError(
                    f"gpype.{name} needs a package that is not installed. "
                    f"{hint}"
                ) from e
            raise AttributeError(f"Cannot import '{name}' from gpype: {e}")

    raise AttributeError(f"module 'gpype' has no attribute '{name}'")


#: Third-party top-level module -> the extra that provides it. Keyed by
#: the module rather than by the g.Pype class, because the module is what
#: the ImportError actually names: a user who asked for TimeSeriesScope
#: and got "No module named 'PySide6'" is owed the sentence that connects
#: the two, not a guess based on which class they happened to ask for.
_EXTRA_FOR_MODULE = {
    "PySide6": "gui",
    "PySide6_Addons": "gui",
    "PySide6_Essentials": "gui",
    "shiboken6": "gui",
    "pyqtgraph": "gui",
    "pynput": "devices",
    "objc": "devices",
    "Quartz": "devices",
    "gtec_ble": "devices",
    "gtec_gds": "devices",
    "gtec_pp": "devices",
    "pylsl": "lsl",
    # h5py answers for both .mat and .h5: MATLAB v7.3 is HDF5 behind a
    # userblock, so one module carries two of the three formats and a
    # user who asked for MatWriter is told about 'formats', not about
    # HDF5.
    "h5py": "formats",
    "pyedflib": "formats",
}


def _missing_extra_hint(error: BaseException) -> str:
    """How to install what *error* says is missing, if an extra has it.

    Since 4.0.0 the frontend, the device drivers and LSL are extras, so
    ``pip install gpype`` no longer carries them -- a deliberate change,
    because a headless server imports none of them and pip cannot be told
    to skip a declared dependency. The cost of that change is this
    message: without it, asking for a scope reports a missing PySide6 and
    leaves the reader to work out that an extra exists at all.

    Args:
        error: The ImportError or AttributeError that was raised.

    Returns:
        str: A sentence naming the extra and the command, or "" when the
            failure has nothing to do with an extra -- in which case the
            original error is the more useful thing to show.
    """
    # `error.name` rather than `getattr(error, "name", None)`: this
    # module's own `getattr` is patched by parts of the suite, and a
    # helper on the error path must not depend on a builtin the caller
    # may have replaced.
    try:
        missing = error.name
    except AttributeError:
        return ""
    if not missing:
        return ""
    extra = _EXTRA_FOR_MODULE.get(str(missing).split(".")[0])
    if extra is None:
        return ""
    return (
        f"{missing!r} is provided by the '{extra}' extra, which "
        f"`pip install gpype` no longer installs on its own:\n"
        f"    pip install 'gpype[{extra}]'\n"
        f"or `pip install 'gpype[all]'` for everything, which is what "
        f"`pip install gpype` gave you before 4.0.0."
    )


def __dir__():
    """Return all available attributes for autocomplete."""
    return list(globals().keys()) + list(_LAZY_IMPORTS.values())
