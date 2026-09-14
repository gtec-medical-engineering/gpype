# Changelog

## [4.0.0] - 2026-09-14

A large release. g.Pype now runs offline as well as in real time, records
to four file formats, drives g.USBamp, supports Python 3.10 to 3.14 on
Windows, macOS and Linux, and installs in layers so a headless server no
longer pulls in a GUI. It moves to ioiocore 5.0.0, whose chain layer backs
every composite source, sink and widget.

### Breaking

- **The install is layered.** `pip install gpype` is now what a server
  needs and carries **no GUI**. `pip install gpype[all]` is what a plain
  install gave before. The extras are `gui` (scopes and widgets),
  `devices` (amplifiers and `Keyboard`), `lsl` and `formats` (the
  `.mat`, `.h5` and `.edf` readers and writers).
- **Linux is supported.** g.Pype runs on Windows, macOS and Linux, and a
  release ships manylinux wheels. `GNautilus`, `GHIamp` and `GUSBamp` no
  longer refuse to construct off Windows -- their driver publishes Linux
  and macOS wheels. The Unicorn Hybrid Black and the Paradigm Presenter
  remain Windows-only.
- Requires ioiocore 5.0.0; g.Pype will not run on ioiocore 4.x.
- `Pipeline.start()` refuses a pipeline that could not work, naming the
  node. Such a pipeline used to start and show a scope that never updated.
- The clocked acquisition path is gone and synchronisation is always on:
  `sync=`, `buffer_delay_ms=`, `output_timeline=`, `output_buffer_level=`
  and the `buffer_level` port no longer exist. Drop them.
- `Hold` is removed — `Interpolator` replaces it. Electrode impedance
  monitoring, `Source.delay` and `Scope`'s `update_rule` are removed.
- `BCICore8` is renamed `BCICore`; the old name still works and warns. An
  eight-channel recording is labelled `Fz C3 Cz C4 Pz PO7 POz PO8` unless
  a `montage` says otherwise.
- A distributed pipeline no longer exchanges data over an unencrypted
  socket unless it is told to.

### Added

- Support for Python 3.14, and wheels for Linux (manylinux_2_28 x86_64)
  alongside Windows and macOS.
- g.USBamp, and OSCAR on all five amplifiers.
- **Offline processing.** `Pipeline.run()` drives a whole recording
  through the pipeline and returns; `gp.Collector` and `gp.Result` hand
  back the data with its rate, channel labels and roles; the filters
  accept `phase="zero"` for zero group delay; and `@gp.node` runs a plain
  function as a node.
- **Recording to MATLAB v7.3, HDF5 and EDF+** and back, through the
  `formats` extra. `.mat` and `.h5` are exact, `.edf` is quantised to its
  physical range. `gp.GtcReader` reads `.gtc` recordings read-only.
- **Record a run and replay it.** `save_as` writes every source's frames
  unmodified; `load_from` replays them on the same block boundaries, in a
  realtime or a batch pipeline. Marker placement is reproduced exactly on
  Windows and Linux; on macOS the first marker of a replay may land a few
  samples away from where the live run put it.
- **Watch a running pipeline.** `Pipeline.attach_probe(node, port)`
  publishes what one output port emits, live, to a subscribing client.
- Ten nodes: `Reference`, `RollingStatistic`, `Threshold`,
  `ChannelSelector`, `ChannelLabeler`, `Marker`, `Baseline`,
  `EpochAverage`, `BandPower` and `CsvReader`.
- `gp.Montage` for naming EEG channels, and a per-channel description
  every source declares: `channel_roles` says how a channel must be
  treated, `channel_labels` says what it is.
- `Pipeline.close()` and context-manager support; `gpype.API_VERSION` and
  `Pipeline.get_nodes()` for tools driving g.Pype from outside; and
  `gp.Chain`, `gp.IChain`, `gp.OChain`, `gp.IOChain` so a user can build a
  composite node the way g.Pype's own are built.
- Season 6 of the g.Pype Training, Rough Waters: troubleshooting and
  pipeline monitoring, in five episodes.

### Fixed

- Marker placement and sample timestamps no longer depend on the Python
  version, and an event arriving in the first frames of a run is placed
  rather than discarded.
- A recording replays at the speed it was asked for. `CsvReader`,
  `EDFReader`, `HDF5Reader` and `MatReader` ran `frame_size` times too
  fast whenever a frame size above 1 was set.
- Scopes repaint at the rate they were configured for, and
  `TimeSeriesScope` redraws roughly three times faster for the same CPU.
- A recording made with a g.tec amplifier present is no longer marked as
  though none was.
- Stopping a pipeline containing `UDPReceiver` no longer risks crashing
  the process. The socket was closed while the listener thread was
  still waiting on it, which on Windows is an access violation rather
  than an exception.
- Many smaller corrections across acquisition, serialisation, the
  distributed link, the scopes and the file readers.

## [3.0.9] - 2026-02-20

We integrated the Unicorn Hybrid Black into g.Pype, improved various performance aspects, and fixed various minor bugs.

- HybridBlack node available. See g.Pype Training season 3 episode 4.
- Enabled direct execution of step() functions of of consecutive nodes to avoid thread spamming.
- Removed PerformanceMonitor widget (not accurate enough).
- Fixed some minor bugs.

## [3.0.8] - 2026-01-15

We fixed a bug in the UDPReceiver node.

- Fixed a set/reset race condition in the UDPReceiver node
- Fixed some bugs in the de/serialization procedure

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
