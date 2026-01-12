# Changelog

## [3.0.7] - 2026-01-12

We fixed a bug in the Router and GenericFilter nodes and updated the file writing mechanism. Also, the Equation node can now handle matrix operations.

- Fixed a bug in synchronous/asynchronous signal propagation in the Router node.
- Ensured numerical stability in GenericFilter nodes by avoiding FIR filters to be forced into biquad structures.
- Split FileWriter class into a FileWriter base class and CsvWriter concrete class, to cleanly enable other formats in the future.
- Updated the Equation node to accommodate matrix operations.
- Minor changes in documentation

## [3.0.6] - 2025-12-10  YANKED

This version contains a serious bug in the propagation of synchronous and asyncronous signals in the Router node, which has been fixed in version 3.0.7. Do not use this version.

Original changelog text:

We applied some small fixes in code and documentation and updated the FileWriter and Router node.

- Updated FileWriter node to store timestamps instead of sample index
- Improved performance of Router node
- Minor fixes in documentation

## [3.0.5] - 2025-10-07

We added Season 2 Episode 2 (Routing Signals) to the g.Pype Training, updated the documentation and switched the theme to dark. We also fixed some minor bugs in the g.Pype source code.

- Migrated documentation to all black design
- Fixed division by zero error in TimeSeriesScope
- Router training page added
- Changed Router input parameters from {input, output}_selector to {input, output}_channels

## [3.0.4] - 2025-09-16
- Updated unit tests
- Updated documentation
- Fingerprints fixed

## [3.0.3] - 2025-08-04
- BCI Core-8 source buffering optimized
- Updated build procedure

## [3.0.1] - 2025-07-17
- Small bugfix in examples
- New build procedure

## [3.0.0] - 2025-07-15
- Moved from metadata to contexts
- Added various nodes (Framer, Decimator, ...)
- Implemented frames
- Implemented multirate support

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
