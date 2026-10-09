# g.Pype

[![Powered by g.tec](https://img.shields.io/badge/powered_by-g.tec-003e6b)](http://gtec.at)
[![PyPI](https://img.shields.io/pypi/v/gpype.svg?label=version&color=003e6b)](https://pypi.org/project/gpype/)
[![Python](https://img.shields.io/pypi/pyversions/gpype.svg?color=003e6b)](https://pypi.org/project/gpype/)
![Tests](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgtec-medical-engineering%2Fgpype%2Fbadges%2Ftests.json)
![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgtec-medical-engineering%2Fgpype%2Fbadges%2Fcoverage.json)
[![License](https://img.shields.io/badge/license-GNCL-003e6b)](https://github.com/gtec-medical-engineering/gpype/blob/main/LICENSE-GNCL.txt)
[![Documentation](https://img.shields.io/badge/doc-gpype.gtec.at-003e6b)](https://gpype.gtec.at/)

g.Pype is a Python Software Development Kit (SDK) for building neuroscience and Brain-Computer Interface (BCI) applications. It is designed to be simple to use, with a clear and well-documented coding interface with many examples that help you get started quickly. It provides essential building blocks that can be combined and adapted to your needs, while remaining open to integration with other Python packages. g.Pype runs pipelines in real time or offline over a recording, on Windows, macOS and Linux.


# Quickstart

Install `gpype` and clone the GitHub repository:

```shell
pip install "gpype[all]"
git clone https://github.com/gtec-medical-engineering/gpype.git
```

g.Pype supports Python 3.10 to 3.14. On Windows with Python 3.13 or 3.14, `gpype[all]` leaves out EDF support, because its package, pyedflib, publishes no wheel there; `pip install "gpype[formats]"` adds it where Microsoft C++ Build Tools are installed.

Navigate to the subfolder `./gpype/examples` and run the example scripts directly from your IDE (e.g., VS Code, PyCharm, ...).

# Supported devices and platforms

| Device | Connection | Platforms |
| --- | --- | --- |
| BCI Core-4, BCI Core-8, gCore | Bluetooth Low Energy | Windows, macOS, Linux (x86_64, aarch64) |
| Unicorn Hybrid Black | Bluetooth | Windows, macOS, Linux |
| g.Nautilus | wireless, through the GDS service | Windows, macOS, Linux |
| g.USBamp | USB, through the GDS service | Windows, macOS, Linux |
| g.HIamp | USB, through the GDS service | Windows, macOS, Linux |

Any other device joins through Lab Streaming Layer or UDP beside a g.tec amplifier, and a pipeline runs on the built-in signal generator with no hardware at all. The full table, with channels, rates and extras, is in the manual's [Supported Devices and Platforms](https://gpype.gtec.at/content/1_basic_concepts/supported_devices.html); measured timing accuracy on every amplifier and platform is on its [benchmarks page](https://gpype.gtec.at/content/4_advanced_topics/benchmarks.html).

# Documentation
Full documentation is available at [gpype.gtec.at](https://gpype.gtec.at).

# License
`gpype` is licensed under the **g.tec Non-Commercial License (GNCL)**. See the [LICENSE](https://github.com/gtec-medical-engineering/gpype/blob/main/LICENSE-GNCL.txt) file for details.
