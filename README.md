# g.Pype

[![Powered by g.tec](https://img.shields.io/badge/powered_by-g.tec-blue)](http://gtec.at)
[![PyPI](https://img.shields.io/pypi/v/gpype.svg?label=PyPI%20version&color=brown)](https://pypi.org/project/gpype/)
[![Python](https://img.shields.io/pypi/pyversions/gpype.svg)](https://pypi.org/project/gpype/)
![Tests](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgtec-medical-engineering%2Fgpype%2Fbadges%2Ftests.json)
![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgtec-medical-engineering%2Fgpype%2Fbadges%2Fcoverage.json)
[![License](https://img.shields.io/badge/License-GNCL-red)](https://github.com/gtec-medical-engineering/gpype/blob/main/LICENSE-GNCL.txt)
[![Documentation](https://img.shields.io/badge/Documentation-GitHub%20Pages-green)](https://gtec-medical-engineering.github.io/gpype/)

g.Pype is a Python Software Development Kit (SDK) for building neuroscience and Brain-Computer Interface (BCI) applications. It is designed to be simple to use, with a clear and well-documented coding interface with many examples that help you get started quickly. It provides essential building blocks that can be combined and adapted to your needs, while remaining open to integration with other Python packages. g.Pype runs on Windows and macOS.


# Installation
`gpype` is available on PyPI:

```
pip install gpype
```

# Documentation
Full documentation is available at [GitHub Pages](https://gtec-medical-engineering.github.io/gpype).

# License
`gpype` is licensed under the **g.tec Non-Commercial License (GNCL)**. See the [LICENSE](https://github.com/gtec-medical-engineering/gpype/blob/main/LICENSE-GNCL.txt) file for details.

# Changelog

## [3.0.3] - 2025-08-04
- BCI Core-8 source buffering optimized
- updated build procedure

## [3.0.1] - 2025-07-17
- small bugfix in examples
- new build procedure

## [3.0.0] - 2025-07-15
- moved from metadata to contexts
- added various nodes (Framer, Decimator, ...)
- implemented frames
- implemented multirate support

## [2.1.2] - 2025-05-05
- Added support for Python 3.8-3.13
- Added support for macOS
- Added basic extra nodes
- Refactored filter implementations into distinct categories (Arithmetic, Delay, LTI, Nonlinear).
- Added `SineGenerator` source node and example.
- Significantly expanded test coverage across modules.
- General improvements and updates across backend, frontend, examples, and configuration.

## [2.1.1] - 2025-04-30
- Minor bugfixes

## [2.1.0] - 2025-04-30
- Bugfixing
- g.Nautilus integrated (Win only)
- ParadigmPresenter integrated (Win only)

## [2.0.0] - 2025-02-26
- First public beta release
