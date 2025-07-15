import sys

# get version
from .__version__ import __version__

# core
from .backend.core.node import Node
from .backend.core.i_node import INode
from .backend.core.o_node import ONode
from .backend.core.io_node import IONode
from .backend.core.i_port import IPort
from .backend.core.o_port import OPort

# filters
from .backend.filters.bandpass import Bandpass
from .backend.filters.bandstop import Bandstop
from .backend.filters.lowpass import Lowpass
from .backend.filters.highpass import Highpass
from .backend.filters.moving_average import MovingAverage

# flow
from .backend.flow.framer import Framer
from .backend.flow.router import Router
from .backend.flow.trigger import Trigger

# sinks
from .backend.sinks.file_writer import FileWriter
from .backend.sinks.lsl_sender import LSLSender
from .backend.sinks.udp_sender import UDPSender

# sources
from .backend.sources.bci_core8 import BCICore8
if sys.platform == "win32":
    from .backend.sources.g_nautilus import GNautilus
from .backend.sources.generator import Generator
from .backend.sources.keyboard import Keyboard
from .backend.sources.udp_receiver import UDPReceiver

# timing
from .backend.timing.decimator import Decimator
from .backend.timing.delay import Delay
from .backend.timing.hold import Hold

# transform
from .backend.transform.equation import Equation
from .backend.transform.fft import FFT

# widgets
from .frontend.widgets.trigger_scope import TriggerScope
if sys.platform == "win32":
    from .frontend.widgets.paradigm_presenter import ParadigmPresenter
from .frontend.widgets.performance_monitor import PerformanceMonitor
from .frontend.widgets.spectrum_scope import SpectrumScope
from .frontend.widgets.time_series_scope import TimeSeriesScope

# common
from .common.constants import Constants
from .common.settings import Settings

# top-level
from .frontend.main_app import MainApp
from .backend.pipeline import Pipeline

# add gpype as preinstalled module
import ioiocore as ioc
ioc.Portable.add_preinstalled_module('gpype')
