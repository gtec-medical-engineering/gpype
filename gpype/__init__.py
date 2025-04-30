import sys

# get version
from .__version__ import __version__

# allow lazy loading
from .backend.core.node import Node
from .backend.core.i_node import INode
from .backend.core.o_node import ONode
from .backend.core.io_node import IONode
from .backend.core.i_port import IPort
from .backend.core.o_port import OPort

from .backend.pipeline import Pipeline
from .backend.filters.lti_filter import LTIFilter
from .backend.sources.base.amplifier_source import AmplifierSource
from .backend.sources.base.event_source import EventSource
from .backend.sources.base.fixed_rate_source import FixedRateSource
from .backend.sources.base.source import Source
from .backend.sources.noise_generator import NoiseGenerator
from .backend.sources.bci_core8 import BCICore8
if sys.platform == "win32":
    from .backend.sources.g_nautilus import GNautilus
from .backend.sources.keyboard_capture import KeyboardCapture
from .backend.sources.udp_receiver import UDPReceiver
from .backend.sinks.file_sink import FileSink
from .backend.routing.router import Router
from .backend.misc.sq_estimator import SQEstimator

from .frontend.main_app import MainApp
from .frontend.widgets.time_series_scope import TimeSeriesScope
from .frontend.widgets.performance_monitor import PerformanceMonitor
if sys.platform == "win32":
    from .frontend.widgets.paradigm_presenter import ParadigmPresenter
    from .frontend.widgets.impedance_chart import ImpedanceChart

from .common.constants import Constants

# add gpype_essentials as preinstalled module
import ioiocore as ioc
ioc.Portable.add_preinstalled_module('gpype')
