# Changelog

## [4.1.0] - 2026-10-09

Every source, sink and widget is now the node itself: the pipeline
builds the chain around it, and a custom one is a single class. One
serialized pipeline can assign each of them to an edge with `edge_id`,
and each edge builds only its own part. Events that arrive before a device's
first samples are no longer lost, and a server takes each stream from
one sender. A frozen application runs only with a g.tec device attached,
a licence lifts the non-commercial mark, and `gp.deployment_report()`
tells a build tool what the licence gate would grant.

### Changed

- The loose `example_*.py` scripts are gone; every example is a
  training episode, `examples/sXeY_<name>.py`.
- *Real-Time Performance* gives measured figures per amplifier and
  platform, with the best and the worst cell marked, and the
  two-machine result for Windows, macOS and Linux.
- The SDK reference lists `Node`, the public functions and decorators,
  and the exceptions.
- The training's screenshots are rendered for high-density screens,
  and the amplifier and paradigm episodes show real recordings.
- Season 4, Episode 10 trains on the window a live `Trigger` cuts and
  uses shrinkage LDA.
- The manual carries the new g.Pype SDK logo.
- The manual gains pages a buyer looks for: Supported Devices and
  Platforms, EEG Data in Python, Data Structures, Threads and
  Propagation, Benchmarks, File Formats, LSL Integration, MNE-Python
  Workflow, BIDS Export and Machine Learning, a *Recent Changes* FAQ
  group, a sitemap, `llms.txt` and redirects from the 3.x URLs.
- **Season 8 of the training is rewritten in the same template**, and its
  Episode 2 is titled *Serialized Pipelines, the Allow-List and Shipping
  Your Node*.
- **Season 7 of the training is rewritten in the same template, with a
  new Episode 6, *A Function as a Node***; the loose custom-node example
  is gone, since Episode 1 teaches the same node.
- **Season 6 of the training is rewritten in the same template, and its
  Episode 5, *When the Numbers Are Wrong*, is removed** with its example.
- **Season 5 of the training is rewritten in the same template**; its
  example files carry their episode's name, and the three paradigm
  episodes share one scaffold that the oddball and the visual episode
  change in one place each.
- **Season 4 of the training is rewritten in the same template, with
  five new episodes: Receiving UDP, Receiving LSL Streams, Into
  MNE-Python, Into scikit-learn, and Record a Run, Replay It
  Anywhere**; its example files carry their episode's name, and the
  SDK reference gains pages for `Epochs`, `Sklearn` and `Standardize`.
- **Season 3 of the training is rewritten in the same template, with
  two new episodes, g.USBamp and g.HIamp**, so every amplifier the SDK
  ships has one; its example files carry their episode's name.
- **Season 2 of the training is rewritten in one template, Read and
  Reference, with three new episodes** (*The Main Window*, *Processing a
  Recording Offline*, *Zero-Phase Filtering*), one example file per
  episode, and examples that print results rather than summaries.
- **Messages and docstrings say "pipeline" where they said "graph".**
- **`GtcReader` is no longer part of the public API.** It read an
  internal format; a pipeline saved with one still loads.
- **Device attestations are accepted under key `0x0004` only**; a
  signature under `0x0001`, `0x0002` or `0x0003` is refused as retired.
  The `devices` extra requires the drivers that sign with `0x0004`:
  gtec-ble 2.9.4, gtec-gds 1.7.3 and gtec-unicorn 1.0.0. Requires
  gtec_attest 1.0.0.
- **A licensed run is no longer marked for want of an attested
  amplifier.**
- **A replayed recording that carries the non-commercial mark marks the
  run that replays it.**
- **A frozen application runs its pipeline only with a g.tec device
  attached**, unless it only replays recordings.
- **A deployment whose licence cannot be determined runs only with an
  attested g.tec device**; a server pipeline in a browser counts as
  development.
- **A pipeline is refused if started under a residency other than the
  one it was built for.**
- **Requires gtec_oscar 1.3.1**, which collects its own data files when
  an application is frozen with PyInstaller.
- **The `gui` extra accepts PySide6 6.10.1 up to any 6.11 release**
  instead of exactly 6.10.1, so installing it no longer downgrades a
  newer PySide6 another package shares the environment with.
- **A distributed pipeline stamps every stream in the server's reference
  clock**: an edge that builds a source syncs its clock to the server
  once at `start()`, before its first stamp, and raises
  `ClockSyncError` if it cannot within 30 s.
- **`BandPower` now computes power: the sum of the squared spectrum
  amplitudes over each band. Every value it outputs changes**, and a
  threshold tuned on its old output needs tuning again; its output
  records no unit.
- **`Keyboard` emits Windows virtual-key codes on macOS and Linux too**,
  so the arrow keys are 37 to 40 and a recorded code means the same key
  on every platform; a key Windows has no code for emits -1.
- The `devices` extra requires pynput 1.8.1, which `Keyboard` needs under
  X on Linux.
- **`gp.Generator` and every other source, sink and widget is the node
  itself, no longer an ioiocore chain.** The pipeline builds the chain
  around it when it is connected; `node.internal_nodes` is gone, and
  `Pipeline.chain_of(node)` returns the chain.
- **A custom source, sink or widget is one class**: subclass its base --
  `gp.Source`, `gp.Sink`, or `gp.Scope` or `gp.Widget` -- write a
  constructor, `setup` and `step`, and connect it. The pipeline gives it
  the same chain the built-in nodes get.
- **The private `_GeneratorCore`-style classes are gone**; import the
  public name. `GPYPE_WRAP_CORES` is gone too: the pipeline always wraps.
- **A 4.0 serialized pipeline loads unchanged.** One whose sources, sinks or widgets
  share a default name loads, but has to have them named before it can be
  saved again.
- **On a server, a `CsvReader`, `MatReader`, `HDF5Reader` or `EDFReader`
  and a g.USBamp, g.HIamp or g.Nautilus take their shape from
  the stream the edge sends**, and open no file or device. A 4.0 serialized pipeline
  that left the rate, channel count or frame size to the file or the
  device loads there, and a server saving it again writes those three
  settings back as the author left them, as it does a serial or
  g.Nautilus sensitivity left to the device. An `LslReceiver` on a server
  takes its shape from the serialized pipeline.
- **A server no longer builds the writer of a sink**, only the link that
  sends the stream to the edges. It never wrote a file.
- **A serialized pipeline whose bundled code imports a package this host does not
  have is refused when it is loaded**, with every such package named
  (`MissingRequirementsError`), before its bundle is unpacked. It used to
  fail while its nodes were rebuilt, naming the first.
- **`start()` and `run()` refuse an input that takes its timing from its
  connection and is not connected, naming it.** An unconnected
  `Collector` is one; it used to run.

- **g.Pype requires ioiocore below 6**, as well as 5.0.0 or later.
- **`HybridBlack` drives the Unicorn through `gtec_unicorn`**, which the
  `devices` extra now installs: UnicornPy and the Unicorn Suite are no
  longer needed, and the node is no longer Windows-only -- it runs on
  Windows x64, Linux x86_64 and macOS arm64.
- **Opening a Unicorn Hybrid Black requires an active Unicorn Python
  Hybrid Black licence**; without one `start()` raises
  `gtec_unicorn.LicenseError`, which names the product.
- **On macOS, `HybridBlack` needs the main thread to run an event loop
  while it acquires**, as `MainApp.run()` does.
- **A `sampling_rate`, `channel_count` or batch `frame_size` given to a
  `CsvReader`, `MatReader`, `HDF5Reader` or `EDFReader` must agree with
  its file**, or is refused with a `ValueError`; a rate the file records
  no longer yields to one given.
- **On Windows a server's data socket refuses a port another program
  holds where it binds, `127.0.0.1` and `::1` included**, with an
  `OSError` naming the port.

### Fixed

- **`BCICore` without a serial no longer opens a Unicorn Hybrid Black**
  in range instead of the gCore or Core-8 beside it.
- `TriggerScope` labels its time axis in milliseconds, and its curves
  are coloured to read on a dark theme.
- A scope freed while the app runs no longer crashes or freezes it
  when a pipeline thread happens to free it.
- **A g.Nautilus pipeline started a second time is attested even when
  the device service briefly does not list the amplifier**; a first open
  still reports an absent g.Nautilus at once.
- **An empty text field's hint is readable on the dark theme**; it was
  near-black on the dark background.
- **Every g.Pype window carries its application's icon, and on Windows
  a script's windows take their own taskbar identity instead of
  Python's**; `MainApp(icon=...)` names the icon, a build names its own,
  else it is g.Pype's.
- **g.Pype's own icon replaces g.tec's** on every window that names
  none, and on the licence tools.
- **A gCore switched off mid-run is reported as an error after 60 s**, as
  a BCI Core-8 is; it read as a dropped link and warned for as long as
  anyone watched.
- **A gCore run no longer opens with "No data for 2.0 s" after its channel
  mode is switched**: a run is given 5 s for its first sample before the
  silence is reported, and a stream that stops still warns after 2 s.
- **A gCore amplifier on firmware 0.2.1.1 or later delivers data again**:
  gtec-ble 2.9.2 is the first release that reads that firmware's
  packets; under 2.9.1 the link stayed connected and the stream empty.
- **`BCICore` recognises a gCore's name** as it does a Core-8's, so it
  opens the gCore rather than another Bluetooth device in range.
- **`pip install "gpype[all]"` no longer fails on Windows with Python 3.13
  or 3.14** for want of a C compiler: it leaves out pyedflib there, which
  `gpype[formats]` still installs where Microsoft C++ Build Tools are
  present.
- **`MainApp` names the `gui` extra when PySide6 or pyqtgraph is
  missing**, before it builds anything; it raised a bare `No module named
  'PySide6'`, and without pyqtgraph it failed later, inside a scope.
- **A message naming a missing extra now quotes its install command with
  double quotes**, `pip install "gpype[gui]"`, so it can be pasted into
  Windows' Command Prompt as well as PowerShell and a POSIX shell.
- **The install commands in the README, the manual and the examples are
  quoted the same way**, so `pip install "gpype[all]"` also works in zsh,
  the default shell on macOS, which refused the unquoted brackets.
- **A scope that is never added to `MainApp`, or is dropped while the app
  runs, no longer crashes the process when Python frees it.**
- **`SpectrumScope` draws the spectrum an `FFT` emits**; since 4.0.0 it
  drew nothing there.
- **A trigger channel crosses LSL as a trigger when the channels have no
  names**: `LSLSender` writes its type, and `LslReceiver` keeps types and
  units whether or not every channel is named.
- **The `TriggerScope` documentation says it shows the average of the
  epochs**, which it always did, not each epoch overlaid.
- **The manual says `Keyboard` needs Input Monitoring on macOS**, not
  Accessibility; without it no key arrives, and nothing reports an error.
- **`start()` refuses a batch pipeline and names `run()`**; it used to
  report the pipeline running and process nothing.
- **Placement warnings reach the session log and the pipeline's
  condition** -- two events in one sample, an event too late, events lost
  at stop -- as well as the console.
- **`UDPReceiver` says why it cannot bind its port**: on macOS and Linux
  the default, 1000, needs root, and it no longer blames another process
  for that.
- **Transport warnings reach the session log and the pipeline's
  condition** -- a stream refused because another edge sends it, a lost
  connection, data dropped while disconnected, a `Link` that timed out
  waiting for the other side -- as well as the console.
- **`EDFWriter` stores a trigger channel unscaled**, so its integer codes
  read back exactly and the events `EDFReader` derives from it carry the
  codes that were emitted.
- **`CsvReader` reads the non-commercial mark `CsvWriter` writes** and
  exposes it as `mark`; a node fitted on a marked CSV is marked.
- **`BandPower`'s output now declares the rate its rows actually arrive
  at, not the signal rate it inherited from `FFT`.** `Result.rate`,
  `times` and `duration` behind it read up to 125 times too fast and
  too short; a `BandPower` fed a context with no frame rate now
  declares none rather than the wrong value.
- **An event is placed on the sample it happened on when the signal's
  frames reach the placing node in bursts** -- after a gap, behind a
  stalled node, or as a late device's or replay's first frames arrive at
  once; it used to land up to hundreds of milliseconds early, with no
  warning.
- **A `Trigger` connects to a `TriggerScope` on a server and on an edge**,
  single-plot or with a port per plot; since 4.0.0 that was refused there.
  Its inputs are `ASYNC` now, so a continuous stream wired straight into a
  new one is refused in every residency.
- **A placing node says when it drops events**: the events still waiting
  to be placed when the pipeline stops, and a late event that finds its
  sample taken, which the late warning called "not dropped".
- **`FFT`, `BandPower`, `Trigger`, `EpochAverage` and `Framer` no longer
  forward a stale marker or gap position onto output rows that no
  longer mean the same sample**; each warns, naming itself. `Delay`
  moves markers and gaps by its delay instead of leaving them in
  place, and drops any that run off the end.
- **A licensed deployment on a read-only mount reads its licence
  instead of refusing to start**, and one whose licence store cannot be
  read is told what to fix. Requires gtec_licensing 2.3.0.
- **`gp.Settings` works where the home directory is missing or cannot be
  written**, such as on Android: it falls back to the temp directory, or
  to memory, and warns once saying which.
- **A g.USBamp, g.HIamp or g.Nautilus pipeline started a second time is
  no longer marked non-commercial.**
- **A licensed BCI Core run is no longer marked non-commercial, and a
  gCore amplifier left to its own rate times the pipeline at that rate.**
- **A licensed Unicorn Hybrid Black run is no longer marked
  non-commercial**: the Unicorn attests.
- **A `HybridBlack` `frame_size` larger than one read of the device can
  hold is refused when the device is opened**, naming the limit, rather
  than failing three reads later.
- **A deployment whose licence cannot be verified is told what to fix**,
  such as setting `GTEC_LICENSE_MACHINE_ID` on Linux or macOS, rather than
  to check its connection to the licensing service.
- **A start refused for licensing no longer holds the amplifier until
  `close()`**; it is free for the next run or process as soon as `start()`
  raises.
- **An edge running a `Marker`, `Keyboard` or `UDPReceiver` no longer
  fails at start** with `ValueError: dtype object cannot be carried in a
  frame`.
- **Events emitted before a device's first samples are no longer lost.**
  A `Marker`, `Keyboard` or `UDPReceiver` event that arrives before the
  master source has delivered anything is placed once it has; one still
  held when the pipeline stops is reported.
- **Stopping a pipeline no longer misreports an event it was placing.**
  An event being placed as the pipeline stopped could be delivered and
  still reported lost, or lost with nothing said, however soon the
  pipeline was started again.
- **A replay places each event on the sample its live run did**, unless
  the event is mapped more than 256 device frames after the frame that
  reaches it (about 4 s at 250 Hz in frames of 4); that one gets the
  timing of the oldest frame still remembered. On macOS the first event
  of a replay can still land a few samples from where the live run put
  it. A `Marker`, `Keyboard` or `UDPReceiver` event is mapped with the
  timing of the first device frame that reaches it, so its sample no
  longer depends on thread timing. A node that reads the events alone
  receives each once the device's samples reach it: up to a frame
  later, later still behind OSCAR's delay, and only once the device's
  stream resumes if it is out.
- **A `Router` selection written as a `range`, a tuple or a numpy array
  now works, and the pipeline serialises.** `json.dumps(p.serialize())`
  used to raise `TypeError: Object of type range is not JSON
  serializable`. The same holds for `Reference`, the scopes'
  `hidden_channels`, a `TimeSeriesScope.Markers` channel and g.USBamp's
  `bipolar_channels`.
- **A server takes each stream from one sender.** Two edges sending the
  same stream -- a source without an `edge_id`, run by both -- were merged
  into one stream at twice its rate; the server now keeps the first and
  warns about the second. A stream a client outside g.Pype feeds, such as
  a browser driving a `Marker`, needs that node assigned an `edge_id` no
  g.Pype process runs as.
- **`UDPReceiver` no longer emits a 0 when it stops** without a trigger
  to reset.
- **A serialized pipeline whose file has changed since it was saved loads with a
  warning** that names the reader and the line that loaded the
  serialized pipeline, and takes the file's shape. A reader that refuses its
  arguments closes its file.
- **A `CsvReader`, `MatReader`, `HDF5Reader` or `EDFReader` serialized pipeline
  pointed at another file runs on that file's shape**, with a warning
  naming what the serialized pipeline stored. It failed with `channel_roles has 1
  entries but the port declares 4 channels`.
- **A long CSV recording reads back at its rate**: 400 s at 250 Hz read
  back as 250.137 Hz.
- **A `.gtc` marker about the whole recording has `channel` None**, not
  4294967295.
- **A BCI Core recording names its class `BCICore`, not `BCI`**, in its
  manifest and in the file an unnamed stream is filed under. Recordings
  made before still replay.

- **A pipeline holding a `GenericFilter` can be written as a
  serialized pipeline:** its `b` and `a` are stored as lists of float, whatever
  they are given as. Numpy arrays made `json.dumps(p.serialize())`
  raise `TypeError: Object of type ndarray is not JSON serializable`.
- **The manual no longer says every reader takes the same arguments**:
  `HDF5Reader` and `MatReader` add `variable_name` and `has_time_row`.
- **A `Result` no longer describes channels a node removed.** After a
  `ChannelSelector`, `Result.units` still listed the unit of every input
  channel; a per-channel list that does not match the channel count is
  now `None`.
- **`ChannelSelector` and `RollingStatistic` describe only the channels
  they emit.** A `.mat` or `.h5` file behind a `ChannelSelector` recorded
  the input's units, at its width and in its order; a `RollingStatistic`
  over an input with a trigger kept the trigger's label, role and unit.
- **`Decimator`, `Interpolator` and `RollingStatistic` keep an exact
  sampling rate exact, `start_time` on their first output sample, and
  markers and gaps on their own samples.** The exact rate passed
  through unchanged, and was written to `.mat` and `.h5` files beside a
  different rate. `start_time` stayed at the input's first sample, 16 ms
  early after `Decimator(decimation_factor=5)` at 250 Hz. Markers and
  gaps kept the input's sample positions. A marker now moves to the first
  output sample at or after it, with its duration rounded up, and a gap
  covers every output sample computed from a missing one.
- **Units stay with their channels behind a `Router`, a `BandPower`, a
  bipolar `Reference`, a node with several inputs and an event stream.**
  A `.mat` or `.h5` file behind them could record another width's units,
  or the input port names as units. An `Equation` records no unit unless
  given one.
- **A g.Nautilus whose headset is switched off now stops with an error.**
  The base station goes on streaming the last sample, held, and the
  pipeline used to stay `Healthy`; the device's validation indicator is
  now watched -- a warning after 2 s of invalid samples, `Error` after
  60 s, so a headset carried out of range and back is not cut off. It is
  added to the output only when asked for.
- **A Unicorn Hybrid Black no longer warns "Falling behind the amplifier"
  while keeping up.** Bluetooth delivers in bursts, and six ready reads in
  a row -- one burst -- raised the warning; it now takes a second of
  backlog.
- **A g.Nautilus can be opened again.** With gtec_gds 1.6.0 every
  `GNautilus(...)` failed with
  `AttributeError: 'GNautilus' object has no attribute 'link_quality'`.
- **Opening a g.Nautilus survives a connect the device service refuses
  once.** "Couldn't open device" is asked again up to twice, two seconds
  apart; any other refusal is reported at once, as before.
- **A g.USBamp, g.HIamp or g.Nautilus that refuses its configuration no
  longer stays claimed.** A refused sampling rate used to leave the device
  locked until the error was discarded, so the next open failed with "An
  already existing data acquisition session cannot be opened
  exclusively".
- The `devices` extra requires gtec_gds 1.7.0, which fixes both of the
  above in the driver itself.
- **CSV recordings keep their timestamps exact.** The time column was
  written to six significant digits, so a recording longer than about 17
  minutes at 250 Hz, or any recording at 256 Hz, stored timestamps coarser
  than a sample and read back at the wrong rate.
- **g.USBamp, g.HIamp and g.Nautilus pipelines can be started again after
  `stop()`.** A second `start()` used to fail with
  `'NoneType' object has no attribute 'start'`.
- **A UDP port Windows has reserved is named as a reservation.**
  `UDPReceiver` reported `WinError 10013` as another process holding the
  port; it now says the port most likely lies in a range Hyper-V or WSL
  reserved, and names the command that lists them.
- **A BCI Core that stops delivering now reports an error, not an endless
  warning.** When the amplifier goes quiet while the Bluetooth link still
  reports itself connected -- which is what a device switched off mid-run
  looks like -- the condition becomes `Error` after 60 seconds and a
  failure entry is recorded. A shorter dropout, such as walking out of
  range and back, stays a warning. A link the driver reports as *down*
  is unchanged: it recovers by itself.
- **A g.USBamp, g.HIamp or g.Nautilus whose acquisition stops mid-run now
  reports an error.** When the driver gives up on the stream -- the
  amplifier switched off, or its cable pulled -- the condition becomes
  `Error` and the failure entry carries the driver's reason. It used to
  stay `Healthy` until `stop()`.
- **`Pipeline.failure` is set as soon as the condition reads `Error`.** It
  used to stay `None` while the pipeline was still stopping, so a poll
  often found no failure; `raise_if_failed()` now names the same entry. A
  `stop()` or `close()` called then waits for that stop instead of
  stopping the nodes a second time, and a `start()` waits for it and then
  restarts instead of being skipped.
- **Building a node no longer prints an argparse usage block in a host
  process whose command line g.Pype does not own.** The implicit read of
  `sys.argv` already fell back to defaults when it could not parse them;
  it now does so silently.
- **Connecting to a BCI Core amplifier no longer fails on one unlucky scan.**
  Discovery scans up to three times instead of once; a healthy amplifier is
  invisible to roughly 6% of Bluetooth scans because it advertises in bursts.
- **`BCICore` names an unusable Bluetooth adapter** when every open attempt
  failed too fast to have reached the amplifier, and names a failed
  usage-key registration as the cause instead when that is what happened.
- **A failed connection now says which of four things went wrong** -- the scan
  could not run, the adapter heard other devices but no amplifier, it heard a
  different amplifier than the serial asked for, or it heard nothing at all --
  instead of always advising that the amplifier be checked.
- **`--load-from` now replays the events recorded beside an amplifier,
  on the samples the live run put them on.** Replaying a run whose
  master source is a `BCICore` delivered the signal correctly and
  silently dropped every `Marker`, `Keyboard`, `UDPReceiver` and
  `LslReceiver` event in the same run; recordings made with earlier
  builds are unaffected and replay correctly on this one.
- **A `BCICore` or `Keyboard` node can be built where its driver package
  is not installed.** The driver is imported when the node starts rather
  than when it is constructed, so a serialized pipeline naming one can be loaded and
  inspected without the `devices` extra; a missing driver is reported at
  `start()`.
- **`gp.Pipeline()` works without psutil installed**, and on platforms with
  no known log directory, such as Android and iOS, where it logs to the
  temp directory.
- **Eight training examples and three manual pages now match what the code
  does**, among them the FFT's output rate, Threshold's parameters and
  Bandstop's parameter names.
- **Every basic example calls `p.close()`**, the keyboard examples no
  longer draw raw key codes over a neighbouring channel, and
  `example_foreign_lsl_receive.py` uses the stream's own sampling rate.
- **The BCI Core example says a Core-4 needs its montage or
  `channel_count=4`**, and the checkerboard/face paradigm example runs
  without hardware, like its siblings.
- **`serialize()` refuses a pipeline holding `Apply` or an `@gp.node`
  node, naming it.** It used to write a serialized pipeline that could not be loaded.
- **The manual no longer says a node parameter not declared in
  `Configuration.Keys` is dropped when a pipeline is saved.** Whatever
  `__init__` passes to the base class is saved; a value kept only on
  `self` is not.
- **Loading a serialized pipeline that carries bundled code no longer leaves a new
  directory in the temporary directory each time**: a bundle is unpacked
  once, reused by every later load, and removed once no process that
  loaded it runs and no load has used it for a week.

### Added

- **The manual's *Real-Time Performance* page gives measured
  performance**: event placement, trigger scatter and load for every
  amplifier on Windows, macOS and Linux, and event placement across
  two machines.
- **`gp.deployment_report()`** says whether the process is a frozen
  application, which licence state it is in for the product the gate
  asks about (g.Pype Runtime unless one is bound), and what the licence
  gate would grant, as a JSON-serialisable dict.
- **`gp.set_licence_product()`** binds an application's own licence
  product, which the licence gate then asks about instead of g.Pype
  Runtime.
- **g.Pype publishes a Pyodide wheel, for Python 3.14 in a browser**:
  `micropip.install("gpype")` installs it, and a server pipeline runs in
  a Web Worker without threads, fed and read by its page, with OSCAR
  (gtec-oscar 1.3.0 or later). psutil, gtec_licensing and gtec_attest are
  not installed there, so a run there is `limited`.
- **In a browser's Web Worker, a SERVER-residency `MainApp.run()` waits
  until the page calls `MainApp.end_run()`**, so the script reaches its
  own `p.stop()` as on a desktop; without JSPI it returns at once, with a
  warning.
- **`BCICore` warns when a gCore's link runs late or drops samples** -- a
  connection interval other than 10 ms, a PHY other than 2M, a sending
  backlog of 100 ms or more, dropped samples -- and notes when each
  clears, with gtec-ble 2.9.3 or later.
- **`get_control`, `set_control`, `invoke_action` and `attach_probe` take
  the node itself** as well as its serialized pipeline id; `gpype.API_VERSION` is
  "1.1".
- **A serialized pipeline can pin what it needs, with `serialize(requirements=[...])`,
  and a deployment that sets `GPYPE_PACKAGE_INDEX` installs every pin it
  lacks when the serialized pipeline is loaded**, from that index only, wheels only,
  into this interpreter; without it, pins do nothing.
- **`Pipeline.restore_requirements()` reinstalls the pins the package
  cache records, with no network**, and `Pipeline.installer_config()`
  reports the setting.
- **g.Pype's own writers refuse a path inside the package cache**, with
  a `ValueError` naming `GPYPE_PACKAGE_CACHE`.
- **`gp.Pipeline.rezero_reference()` zeroes a server's reference clock**,
  which `Pipeline.deserialize` does under server residency; a `.mat` or
  `.h5` recording notes a stream whose clock sync was missing or had not
  converged.
- **`Pipeline.start()` stamps a provenance record into every run**:
  `Result.provenance` gives the serialized pipeline, a content hash, package
  versions and the execution mode, and `Result.document()` recovers the
  serialized pipeline, ready to rebuild and rerun the pipeline.
- **A `.mat` or `.h5` file now records its run's provenance, and
  reading it back carries the lineage forward**: reprocessing a g.Pype
  recording traces back to the run and the input that produced it.
  `.csv` and `.edf` cannot carry it.
- **A fitted node's stale configuration is refused.** Loading or
  starting an artifact whose recorded configuration disagrees with the
  node's current one is refused, naming the node, instead of silently
  applying a fit that no longer matches.
- **`Pipeline.run_all(files)` runs the same pipeline once per file**,
  returning one result per file; it refuses a pipeline holding a
  function node (`Apply`, `@gp.node`) up front, before any file is
  read, and a corrupt file stops the run naming it.
- **`GNautilus`, `GHIamp`, `GUSBamp`, `BCICore` and `HybridBlack`
  declare each channel's physical unit**: EEG in `uV`, and each
  amplifier's own extras -- acceleration, gyroscope, battery, link
  quality, counter -- in the driver's own unit.
- **`ChannelLabeler(units=[...])` declares or overrides a stream's
  units**, the route for a `Generator`, a foreign LSL stream, or any
  source this package does not itself describe.
- **`EDFWriter` writes each channel's unit as its EDF physical
  dimension, and `EDFReader` reads it back**; both used to give every
  channel `'uV'` on write, including a trigger, and nothing on read.
- **`LSLSender` writes each channel's unit into its LSL stream
  description, and `LslReceiver` reads it back.**
- **`Baseline(mode='percent')` declares `'%'`.**
- **`Result.channels` gains `gain`, `offset`, `clipping` and
  `filters`**, GTC's calibration field set, carried through a channel
  selection and round-tripped by `.mat` and `.h5` files.
- **An amplifier's total OSCAR-added delay is readable as
  `oscar_delay_samples`.**
- **`gp.Epochs(condition, tmin, tmax)` cuts a `(time, channel, trial)`
  block from a batch recording's markers**, refusing an epoch that
  straddles a gap or names an unknown condition, and dropping one that
  runs off either end with a warning giving the count.
  `EpochAverage` averages such a block over its trial axis directly, so
  `Epochs -> EpochAverage -> Collector` returns the averaged result in
  one run.
- **`Result.trials` gives each trial's source sample and condition
  label, and `Result.plot()` draws a trial mean or one trial via
  `trial=i`, labelled from it, and marks every event as a labelled
  vertical line.**
- **A `.mat` or `.h5` recording with no events model of its own, but a
  trigger channel, now reads back with derived events**: an onset at
  each change to a new nonzero value, labelled with the value.
- **One serialized pipeline can assign each source, sink and widget to an edge.**
  Give the node `edge_id="A"` and start that edge with `--edge-id A`, or
  with `GPYPE_EDGE_ID=A` in its environment: edge A opens only its own
  devices, writes only its own files and draws only its own scopes, and
  another edge builds none of it. A node without an `edge_id` runs on
  every edge, as before, and a serialized pipeline that assigns none is saved
  unchanged. An edge id given to a server or a standalone run is refused.
- **A distributed run says what its edges are missing.** An edge warns at
  start when it opens a source the serialized pipeline leaves to every edge, or when
  two edge ids differ only in case; 30 s after start, a server warns of
  each node whose edge has not sent or received its stream.
- **`Result.rate_exact` is the sampling rate as an exact fraction**, such
  as `512000/1001` from a `.gtc` recording. `Result.rate` stays a float.
- **`Result.channels` gives the name, role and unit of each channel**,
  and `repr(result)` shows the first channel labels and the trust state.
- **Markers, gaps and trust survive `.mat` and `.h5` files.**
- **`EDFReader` reads EDF+ annotations as `Result.events`, and
  `EDFWriter` writes markers as annotations**, one fewer than the file
  has data records; a marker that does not fit, or names a channel, is
  counted in a warning, and a label longer than 40 bytes is cut and
  named in one.
- **`Result.to_si()` returns the samples in SI units**, and
  `Result.si_scale` and `Channel.si_scale` give each channel's factor. A
  unit that is not recorded, or is not one g.Pype knows, is refused by
  name rather than taken to be microvolts. A
  `RollingStatistic(statistic="var")` records no unit, since a variance
  is in its input's unit squared.
- **`Equation(unit="uV")` names the unit of its output.**
- **`BCICore(sampling_rate=...)` can configure the amplifier instead of
  only checking it.** On a gCore device the rate selects the channel mode
  — 250 Hz/16 channels, 500/8, 1000/4 — and the device is switched into
  it at connect, so `channel_count` need not be given.
- **The attestation verifier trusts each key only for its own kind**: a
  device attestation signed with a licence key is refused, and so is a
  licence attestation signed with an amplifier key.
- **Scans are bounded by the node's `SCANNING_TIMEOUT_S`** (6 s, as
  before) rather than by the driver's module constant, and a BCI Core
  scan that fails on the usage key says whether none was registered or it
  was made for another host.
- **ioiocore 5.1.0 is now the minimum**: a pipeline whose `start()` failed
  can be started again, and one runs where no thread can be started.
- **`gp.Source`, `gp.Sink`, `gp.Widget` and `gp.Scope` are public**: the
  bases a custom source, sink or widget subclasses, no longer imported
  from `gpype.backend` or `gpype.frontend`.
- **`Pipeline.check(document)` makes two of the refusals loading a
  serialized pipeline makes, without loading it**: the allow-list, then the bundled
  code's missing packages. Nothing is unpacked or built, and no module the
  serialized pipeline names is imported. `gp.ModuleNotAllowedError`,
  `gp.MissingRequirementsError` and its base `gp.BundleError` are public.
- **Every file reader publishes the identity of its file, as
  `result.context["input"]`: name, size, mtime and SHA-256.**
- **`gpype.common.document.with_fresh_ids(document)` builds one serialized pipeline
  into several pipelines in one process.**
- **A pipeline and a node print what they are, and a `Result` shows as a
  table in a notebook.**
- **A `Collector` accepts an asynchronous input such as a `Trigger`, and
  stacks the epochs into `(time, channel, trial)`.**
- **The manual gains four episodes and a `Result` reference page, and
  publishes the batch examples:** closing a feedback loop with `Delay`,
  an RMS envelope with `RollingStatistic`, from a run into pandas, and
  a decision in a `ResultScope`; the offline-processing page includes
  the four `example_batch_*.py`, `example_batch_apply.py` among them,
  and Season 4 Episode 5 reads each of its recordings back.
- **A node can fit offline and deploy online.** `Pipeline.fit()` runs a
  batch pipeline once, letting each fittable node learn from the whole
  recording; the fitted state travels in the serialized pipeline, and a realtime
  start refuses a fittable node with no state, or one whose fitted
  shape disagrees with the input, naming it either way. An artifact
  fitted on marked, unsealed or recovered data carries that status
  forward into every run that deploys it.
- **`gp.Standardize` fits each channel's mean and standard deviation
  offline and applies the same z-score online, unchanged.**
- **`gp.Sklearn(estimator)` bridges `LinearDiscriminantAnalysis`,
  `LogisticRegression`, `StandardScaler` or `PCA` into the same fit
  offline, deploy online contract.** scikit-learn is not a dependency;
  constructing this node without it installed names the package to
  install. `example_bridge_sklearn.py` records its own two-class data
  and shows the whole route, offline and deployed.
- **Season 7 of the g.Pype Training, Custom Rigging, is published**:
  your own node, several ports and an asynchronous output, a source on
  `gp.Source` and a sink on `gp.Sink`, a control surface, and a widget on
  `gp.Scope`, in five episodes whose examples run without hardware.
- **The manual gains Season 8, New Horizons, and an MNE bridge**: one
  script run as an edge and a server, and serialized pipelines, the module
  allow-list and bundling your own node; `example_bridge_mne.py` hands a
  recording to MNE-Python in volts and refuses one that records no
  units.

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
- **`Threshold`'s `level`, `release` and `dwell` can be changed while the
  pipeline is running.** It is the first shipped node with a control
  surface: `Pipeline.set_control` moves them without a stop and restart,
  which is what finding a neurofeedback level needs. A refused change
  leaves the node untouched.
- **The node catalogue says which nodes are controllable.**
  `catalog/catalog.json` carries a `controllable` flag per node, so a tool
  composing a document can see the surface before anything runs.

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
