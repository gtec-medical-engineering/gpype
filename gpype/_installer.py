"""Installing a document's pinned requirements, where a deployment opts in.

A document may name the distributions it needs beyond g.Pype, as exact
pins under its ``requirements`` key (D-CORE-109). Where the deployment
has set :data:`ENV_INDEX`, loading the document installs every pin this
environment lacks, from that index only, into this interpreter
(D-CORE-85, D-CORE-113, D-CORE-114). Where it has not, the pins install
nothing and refuse nothing (D-CORE-111).

Plain Python, and not compiled (D-CORE-115): this decides what code is
installed on a customer's host, and a policy a customer can read and
audit is a feature, as :mod:`gpype._transport` says of its own.

The rules, and they are the whole policy:

* **The index is the opt-in.** Read once from the environment, never
  from ``LaunchConfig``, a bind, the allow-list or pip's own
  configuration: every pip call runs ``--isolated``, with every ``PIP_*``
  variable removed and ``PIP_CONFIG_FILE`` set to the null device, and
  ``--extra-index-url`` is never passed (D-CORE-112).
* **Exact pins, and the pins are the closure.** A distribution installed
  here satisfies a requirement as installed; anything else a pinned wheel
  needs must be pinned too (D-CORE-86, D-CORE-110). One environment
  serves every document, so a pin other than the installed version is
  refused, never installed over it (D-CORE-87).
* **A wheel never changes another distribution's files.** Every path its
  install writes is mapped as pip maps it: its root and ``.data``
  purelib and platlib into site-packages; ``.data`` scripts, data and
  headers, and the scripts of its console and GUI entry points, through
  sysconfig. A path another distribution recorded is refused unless it
  already holds the same bytes, and in site-packages and the data
  directory so is another form of one of its import names (``x/``,
  ``x.py``, ``x.<suffix>``). Outside site-packages, so is a path no
  distribution records: the environment's own, such as ``pyvenv.cfg``.
  Not checked: two pins of one document against each other, any other
  directory on ``sys.path``, and what a ``.pth`` file does (D-CORE-119).
* **The cache is no place for a recording.** g.Pype's own file writers
  refuse a path inside it, by name and by file identity; a custom node's
  own code is the allow-list's to trust (D-CORE-120).
* **Wheels only, planned before anything changes.** pip is asked for the
  plan of an install from the absent pins' wheels alone, and the install
  runs only if that plan is exactly them (D-CORE-113).
* **pip never reads the cache.** The cache is a directory others may
  write (D-CORE-88). Each operation copies exactly the wheels it needs
  into a private directory and checks each copy against the sha256
  recorded at its download; pip reads only that directory, and runs in
  an empty one. A download reaches the cache only once its plan has
  passed (D-CORE-117).
* **Install where the interpreter looks, or refuse.** Site-packages must
  be writable, and pip is never let fall back to a user installation
  (D-CORE-118).

Nothing here runs at import: ``import gpype`` never starts pip.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

from .common._private.bundle import BundleError

#: The opt-in, and the only index a document's pins are downloaded from.
ENV_INDEX = "GPYPE_PACKAGE_INDEX"

#: The wheel cache. In a container, the volume (D-CORE-88).
ENV_CACHE = "GPYPE_PACKAGE_CACHE"

#: Where the cache is when :data:`ENV_CACHE` is not set.
DEFAULT_CACHE = os.path.join("~", ".gtec", "gpype", "packages")

#: The first pip with ``--dry-run --report``.
MINIMUM_PIP = (22, 2)

#: How long one pip call may take, in seconds.
PIP_TIMEOUT_S = 600.0

#: How long an install waits for another one to release the cache.
LOCK_TIMEOUT_S = 2 * PIP_TIMEOUT_S

#: The cache's record of every wheel it holds, and its hash.
MANIFEST = "manifest.json"

#: The file whose lock serialises installs across processes.
LOCK_FILE = ".lock"

#: General options every pip call carries.
PIP_OPTIONS = ("--isolated", "--disable-pip-version-check", "--no-input")

#: In a private directory: the wheels pip is given, and pip's empty
#: working directory.
WHEELS = "wheels"
WORKDIR = "cwd"


class RequirementError(BundleError):
    """A pin is not exact, or a pinned wheel needs a distribution nobody
    pinned."""

    def __init__(self, message: str, offenders: Iterable[str] = ()):
        super().__init__(message)
        #: What the message names, redacted.
        self.offenders = list(offenders)


class RequirementConflictError(BundleError):
    """A pin, or what a pinned wheel needs or writes, conflicts with what
    is installed here."""

    def __init__(self, message: str, offenders: Iterable[str] = ()):
        super().__init__(message)
        #: The conflicting pins or requirements.
        self.offenders = list(offenders)


class InstallerError(BundleError):
    """pip, or this interpreter, cannot install a document's pins."""


class _PipCannotRun(InstallerError):
    """pip could not be started: nothing about one entry, so a restore
    stops rather than skipping it."""


class _PipTimedOut(InstallerError):
    """One pip call did not finish in time: a restore skips its entry."""


# ----------------------------------------------
# configuration, read once


@dataclass(frozen=True)
class _Configuration:
    index: Optional[str]
    cache: str


_configuration: Optional[_Configuration] = None


def _config() -> _Configuration:
    """Read the configuration once per process, as the allow-list is."""
    global _configuration
    if _configuration is None:
        index = os.environ.get(ENV_INDEX, "").strip() or None
        cache = os.environ.get(ENV_CACHE, "").strip() or DEFAULT_CACHE
        _configuration = _Configuration(
            index, os.path.abspath(os.path.expanduser(cache))
        )
    return _configuration


def enabled() -> bool:
    """Whether this deployment installs a document's pins."""
    return _config().index is not None


def configuration() -> dict:
    """The opt-in, as JSON-safe values for a control plane to report.

    Returns:
        ``{"enabled": bool, "index": str or None, "cache": str}``, the
        index redacted.
    """
    config = _config()
    return {
        "enabled": config.index is not None,
        "index": None if config.index is None else redact_url(config.index),
        "cache": config.cache,
    }


def _index() -> str:
    """The configured index, refused unless pip would use it as given.

    Plaintext ``http`` is accepted for a loopback host only: pip ignores
    any other insecure index unless told to trust it, and nothing here
    tells it to.

    Raises:
        InstallerError: If the index is not such a URL.
    """
    index = _config().index or ""
    try:
        parts = urlsplit(index)
        host = parts.hostname
    except ValueError:
        parts, host = None, None
    scheme = parts.scheme.lower() if parts else ""
    if scheme == "http":
        from ._transport import is_loopback

        usable = bool(host) and is_loopback(host)
    else:
        usable = scheme == "file" or (scheme == "https" and bool(host))
    if not usable or re.search(r"\s", index):
        raise InstallerError(
            f"{ENV_INDEX} must be the URL of a package index, https (http "
            f"on this machine only) or file; it is "
            f"{redact_url(index)!r}. Nothing was installed."
        )
    return index


# ----------------------------------------------
# the cache, kept from g.Pype's own writers


def _fold(path: str) -> str:
    """*path* normalised as this platform's file systems compare it by
    default: case-insensitively on Windows and macOS."""
    path = os.path.normcase(os.path.normpath(path))
    return path.lower() if sys.platform == "darwin" else path


def _plain(path: str) -> str:
    r"""*path* without a Win32 ``\\?\`` prefix, where it has a plain form:
    ``\\?\C:\x`` is ``C:\x`` and ``\\?\UNC\host\share`` is
    ``\\host\share``. Any other path as given."""
    if os.name != "nt":
        return path
    text = path.replace("/", "\\")
    if text[:8].upper() == "\\\\?\\UNC\\":
        return "\\\\" + text[8:]
    if text[:4] == "\\\\?\\" and text[4:5].isalpha() and text[5:6] == ":":
        return text[4:]
    return path


def _in_by_identity(path: str, directory: str) -> bool:
    r"""Whether *directory* is, by file identity, *path* or a directory it
    lies in.

    For what no comparison of names sees: an administrative share
    (``\\localhost\C$``), a volume GUID or ``\\?\GLOBALROOT`` names the
    same directory under another name. Each ancestor of *path* that
    exists, absolute and resolved, is compared with *directory*. False
    where *directory* does not exist, or its file system reports no file
    identity.
    """
    try:
        wanted = os.stat(directory)
    except (OSError, ValueError):
        return False
    if not wanted.st_ino:
        return False
    seen = set()
    for current in (os.path.abspath(path), os.path.realpath(path)):
        while current not in seen:
            seen.add(current)
            try:
                if os.path.samestat(os.stat(current), wanted):
                    return True
            except (OSError, ValueError):
                pass
            current = os.path.dirname(current)
    return False


def refuse_cache_path(path, what: str) -> None:
    """Refuse *path* as a place to write a recording if it lies in the cache.

    The installer and a document's sinks run as one process user, so no
    file permission keeps a sink out of the cache, and whoever writes the
    cache chooses what a later load or restore installs (D-CORE-117). So
    g.Pype's own writers ask this, with or without the opt-in, since a
    cache filled while it is off is restored once it is on (D-CORE-120).

    Args:
        path: The path about to be written, as configured: absolute, or
            relative to the working directory. ``..`` and symbolic links
            are resolved, and a directory on its way that is the cache
            under another name is found by file identity.
        what: Who writes it, for the message.

    Raises:
        ValueError: If *path* is the cache or lies inside it, naming
            :data:`ENV_CACHE`.
    """
    configured = _plain(_config().cache)
    given = os.fspath(path)
    cache = _fold(os.path.realpath(configured))
    target = _fold(os.path.realpath(_plain(given)))
    if (
        target == cache
        or target.startswith(cache.rstrip(os.sep) + os.sep)
        or _in_by_identity(given, configured)
    ):
        raise ValueError(
            f"{what} may not write {os.fspath(path)!r}: it lies inside the "
            f"package cache, the directory {ENV_CACHE} names (by default "
            f"{DEFAULT_CACHE}), and whoever writes the cache chooses what a "
            f"later load installs. Write the recording elsewhere."
        )


# ----------------------------------------------
# redaction


_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"<>]+")


def _redact_pairs(text: str) -> str:
    """``name=value&...`` with every value replaced by ``***``."""
    return "&".join(
        (item.split("=", 1)[0] + "=***") if "=" in item else "***"
        for item in text.split("&")
    )


def redact_url(url: str) -> str:
    """*url* with its userinfo, query values and fragment redacted.

    A credential in the URL's path is not recognised: a token belongs in
    the userinfo or a query value (D-CORE-112).

    Args:
        url: A URL, as configured or as pip printed it.

    Returns:
        The URL, safe to report.
    """
    try:
        parts = urlsplit(str(url))
    except ValueError:
        return "<a URL that does not parse>"
    netloc = parts.netloc
    if "@" in netloc:
        netloc = "***@" + netloc.rpartition("@")[2]
    query = _redact_pairs(parts.query) if parts.query else ""
    fragment = _redact_pairs(parts.fragment) if parts.fragment else ""
    return urlunsplit((parts.scheme, netloc, parts.path, query, fragment))


def _secrets() -> list:
    """The configured index's credentials, as pip might print them."""
    index = _config().index
    if not index:
        return []
    try:
        parts = urlsplit(index)
        found = [parts.password or parts.username or ""]
    except ValueError:
        return []
    for pairs in (parts.query, parts.fragment):
        found += [item.split("=", 1)[-1] for item in pairs.split("&") if item]
    secrets = set()
    for value in found:
        for form in (value, unquote(value)):
            if len(form) >= 4:
                secrets.add(form)
    return sorted(secrets, key=len, reverse=True)


def redact(text: str) -> str:
    """*text* with credentials removed from every URL it contains.

    Also removes the configured index's own credentials wherever they
    occur, in whatever form pip printed them.

    Args:
        text: A message, or pip's output.

    Returns:
        The text, safe to report.
    """
    text = _URL.sub(lambda match: redact_url(match.group(0)), str(text))
    for secret in _secrets():
        text = text.replace(secret, "***")
    return text


def _shown(value, limit: int = 120) -> str:
    """A value from a document or the manifest, safe and short to report."""
    return redact(repr(value))[:limit]


# ----------------------------------------------
# pins and versions


_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"

#: A PEP 440 version as a pin may spell it: normalised, lower case, no
#: wildcard, no whitespace. Nothing an option, a URL or a marker needs
#: can occur in it.
_PIN_VERSION = (
    r"(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?(?:\.dev[0-9]+)?(?:\+[a-z0-9]+(?:\.[a-z0-9]+)*)?"
)

_PIN = re.compile(rf"(?P<name>{_NAME})==(?P<version>{_PIN_VERSION})", re.ASCII)

#: Longer than any real pin, so a pathological one is refused unread.
_MAX_PIN = 256


def normalize(name: str) -> str:
    """A distribution name as PEP 503 compares it."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Pin:
    """One exact pin, ``name==version``."""

    name: str
    version: str

    @property
    def key(self) -> str:
        """The name, normalised."""
        return normalize(self.name)

    def __str__(self) -> str:
        return f"{self.name}=={self.version}"


def _pin_of(text) -> Optional[Pin]:
    """Parse one pin, or None if *text* is not one."""
    if not isinstance(text, str) or len(text) > _MAX_PIN:
        return None
    match = _PIN.fullmatch(text)
    if match is None:
        return None
    return Pin(match["name"], match["version"])


def parse_pins(requirements) -> list:
    """Parse a document's requirements into pins.

    Args:
        requirements: A list of ``name==version`` strings, or None.

    Returns:
        The pins, in order.

    Raises:
        RequirementError: If *requirements* is not a list, or any entry
            is not an exact pin or names a distribution a second time,
            naming every such entry.
    """
    if requirements is None:
        return []
    if not isinstance(requirements, (list, tuple)):
        shown = _shown(requirements)
        raise RequirementError(
            f"requirements must be a list of exact pins, name==version, "
            f"not {type(requirements).__name__} {shown}.",
            [shown],
        )
    pins, offenders, seen = [], [], set()
    for entry in requirements:
        pin = _pin_of(entry)
        if pin is None or pin.key in seen:
            offenders.append(_shown(entry))
            continue
        seen.add(pin.key)
        pins.append(pin)
    if offenders:
        raise RequirementError(
            f"requirements must be exact pins, name==version, each naming "
            f"one distribution once; {len(offenders)} are not: "
            f"{', '.join(offenders)}. A document names what ran, so a "
            f"range, a wildcard, an extra, a marker, a URL or an option is "
            f"refused rather than resolved.",
            offenders,
        )
    return pins


#: A PEP 440 version however it is spelled, as pip and packaging accept.
_VERSION = re.compile(
    r"""
    ^\s*v?
    (?:(?P<epoch>[0-9]+)!)?
    (?P<release>[0-9]+(?:\.[0-9]+)*)
    (?P<pre>[-_.]?(?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
        [-_.]?(?P<pre_n>[0-9]+)?)?
    (?P<post>(?:-(?P<post_n1>[0-9]+))
        |(?:[-_.]?(?P<post_l>post|rev|r)[-_.]?(?P<post_n2>[0-9]+)?))?
    (?P<dev>[-_.]?(?P<dev_l>dev)[-_.]?(?P<dev_n>[0-9]+)?)?
    (?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

_PRE = {"alpha": "a", "beta": "b", "c": "rc", "pre": "rc", "preview": "rc"}


def _version_key(text: str) -> tuple:
    """A version as PEP 440 compares it for equality.

    Returns:
        ``(public, local)``: two versions are equal when both are, and
        a pin without a local label matches any local label.
    """
    match = _VERSION.match(str(text))
    if match is None:
        return (("legacy", str(text).strip().lower()), None)
    release = [int(part) for part in match["release"].split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    pre = None
    if match["pre_l"]:
        letter = match["pre_l"].lower()
        pre = (_PRE.get(letter, letter), int(match["pre_n"] or 0))
    post = None
    if match["post"]:
        post = int(match["post_n1"] or match["post_n2"] or 0)
    dev = int(match["dev_n"] or 0) if match["dev"] else None
    local = match["local"]
    if local:
        local = re.sub(r"[-_.]", ".", local.lower())
    public = (int(match["epoch"] or 0), tuple(release), pre, post, dev)
    return (public, local)


def satisfies(pinned: str, version: str) -> bool:
    """Whether *version* satisfies ``==pinned``, under PEP 440.

    Args:
        pinned: The version a pin names.
        version: An installed or offered version.

    Returns:
        True if they are equal, a local label on *version* ignored when
        the pin carries none.
    """
    pin_public, pin_local = _version_key(pinned)
    public, local = _version_key(version)
    return pin_public == public and (pin_local is None or pin_local == local)


def installed_version(name: str) -> Optional[str]:
    """The version of *name* this process can import, or None."""
    try:
        return importlib.metadata.version(name) or "unknown"
    except importlib.metadata.PackageNotFoundError:
        return None


def _absent(pins: list) -> list:
    return [pin for pin in pins if installed_version(pin.name) is None]


def check_conflicts(pins: list) -> None:
    """Refuse a pin other than the version installed here (D-CORE-87).

    Args:
        pins: Parsed pins.

    Raises:
        RequirementConflictError: Naming every such pin and the version
            installed.
    """
    conflicts = []
    for pin in pins:
        version = installed_version(pin.name)
        if version is not None and not satisfies(pin.version, version):
            conflicts.append((pin, version))
    if conflicts:
        named = ", ".join(
            f"{pin} (installed here: {version})" for pin, version in conflicts
        )
        raise RequirementConflictError(
            f"this document pins {len(conflicts)} distribution(s) at a "
            f"version other than the one installed here: {named}. One "
            f"environment serves every document this deployment loads, so "
            f"a pin is never installed over another version; pin the "
            f"installed version, or install the pinned one where this "
            f"deployment is built.",
            [str(pin) for pin, _ in conflicts],
        )


# ----------------------------------------------
# running pip


#: ``runner(arguments, timeout_s, cwd)`` returns an object with
#: ``returncode``, ``stdout`` and ``stderr``, as :func:`subprocess.run`
#: does. *cwd* is the empty private directory pip runs in.
Runner = Callable[[list, float, str], object]


def _pip_environment() -> dict:
    """The environment pip runs in: none of pip's own configuration."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PIP_")
    }
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _run(arguments: list, timeout: float, cwd: str):
    """Run one pip command: an argument list, no shell, a timeout.

    In *cwd*, an empty private directory: ``python -m`` puts the working
    directory first on the path pip imports from.
    """
    return subprocess.run(
        arguments,
        shell=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=_pip_environment(),
        cwd=cwd,
    )


#: The runner every pip call goes through. Tests replace it.
_runner: Runner = _run


def _pip(*arguments: str) -> list:
    return [sys.executable, "-m", "pip", *arguments]


def _pip_version() -> Optional[tuple]:
    """The version of this interpreter's pip, or None when it has none."""
    if importlib.util.find_spec("pip") is None:
        return None
    version = installed_version("pip")
    if version is None:
        return None
    match = re.match(r"(\d+)\.(\d+)", version)
    return (int(match[1]), int(match[2])) if match else (0, 0)


def _require_pip() -> None:
    """Refuse where pip cannot install into this interpreter.

    Raises:
        InstallerError: In a frozen interpreter, without an executable,
            or with pip absent or older than :data:`MINIMUM_PIP`.
    """
    if getattr(sys, "frozen", False):
        raise InstallerError(
            "this interpreter is frozen into an application, so pip cannot "
            "install into it. Nothing was installed."
        )
    if not sys.executable:
        raise InstallerError(
            "this interpreter reports no executable to run pip with. "
            "Nothing was installed."
        )
    version = _pip_version()
    if version is None:
        raise InstallerError(
            "pip is not installed in this interpreter, so a document's pins "
            "cannot be installed. Nothing was installed."
        )
    if version < MINIMUM_PIP:
        raise InstallerError(
            f"pip {version[0]}.{version[1]} is older than "
            f"{MINIMUM_PIP[0]}.{MINIMUM_PIP[1]}, the first that can report "
            f"an install's plan before making it. Nothing was installed."
        )


def _site_packages() -> list:
    """Where pip installs for this interpreter: purelib, then platlib."""
    found = []
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        path = os.path.abspath(paths[key])
        if all(os.path.normcase(path) != os.path.normcase(p) for p in found):
            found.append(path)
    return found


def _writable(directory: str) -> bool:
    """Whether this process can create a file in *directory*.

    Tried, not asked: ``os.access`` does not read Windows ACLs.
    """
    try:
        handle, probe = tempfile.mkstemp(prefix=".gpype-probe-", dir=directory)
    except OSError:
        return False
    os.close(handle)
    with contextlib.suppress(OSError):
        os.remove(probe)
    return True


def _require_site_packages() -> None:
    """Refuse where pip could not install where this interpreter imports.

    pip would fall back to a user installation, which this process may
    not import from before it restarts (D-CORE-118); every install also
    passes ``--no-user``.

    Raises:
        InstallerError: Naming each directory that is not writable.
    """
    closed = [path for path in _site_packages() if not _writable(path)]
    if closed:
        raise InstallerError(
            f"this interpreter's site-packages, {', '.join(closed)}, is not "
            f"writable by this process, so a document's pins cannot be "
            f"installed where it imports them. Nothing was installed."
        )


def _output(result) -> str:
    """pip's output, redacted before it is cut, so no URL is cut first."""
    text = f"{result.stdout or ''}{result.stderr or ''}".strip()
    return redact(text)[-4000:]


def _call(run: Runner, arguments: list, what: str, private: str):
    """Run pip once, in *private*'s empty working directory.

    Raises:
        _PipTimedOut: If pip does not finish within :data:`PIP_TIMEOUT_S`.
        _PipCannotRun: If pip cannot be started.
    """
    try:
        return run(list(arguments), PIP_TIMEOUT_S, _workdir(private))
    except subprocess.TimeoutExpired:
        failure = _PipTimedOut(
            f"pip did not finish {what} within {PIP_TIMEOUT_S:.0f} s."
        )
    except OSError as error:
        failure = _PipCannotRun(
            f"pip could not be started for {what}: {redact(error)}"
        )
    # Raised after the handler, so that nothing chains to the exception,
    # whose command line carries the index URL.
    raise failure


# ----------------------------------------------
# the cache, and the private directory pip reads


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_WHEEL = re.compile(
    r"(?P<name>[^\s/\\-]+)-(?P<version>[^\s/\\-]+)(?:-\d[^\s/\\-]*)?"
    r"-(?P<python>[^\s/\\-]+)-(?P<abi>[^\s/\\-]+)-(?P<platform>[^\s/\\-]+)"
    r"\.whl",
    re.IGNORECASE,
)


def _wheel_of(filename: str, pin: Pin) -> bool:
    """Whether *filename* is a plain wheel file name of *pin*."""
    match = _WHEEL.fullmatch(filename)
    return bool(
        match
        and normalize(match["name"]) == pin.key
        and satisfies(pin.version, match["version"])
    )


_supported: Optional[dict] = None


def _supported_tags() -> dict:
    """This interpreter's wheel tags, each mapped to its rank, best first.

    As pip ranks them: from pip's own copy of ``packaging``, else from
    ``packaging``.

    Raises:
        InstallerError: If neither can be imported.
    """
    global _supported
    if _supported is None:
        for name in ("pip._vendor.packaging.tags", "packaging.tags"):
            try:
                tags = importlib.import_module(name)
                break
            except ImportError:
                continue
        else:
            raise InstallerError(
                "neither pip nor packaging says which wheels this "
                "interpreter installs. Nothing was installed."
            )
        ranked = {}
        for rank, tag in enumerate(tags.sys_tags()):
            ranked.setdefault((tag.interpreter, tag.abi, tag.platform), rank)
        _supported = ranked
    return _supported


def _rank(filename: str) -> Optional[int]:
    """How this interpreter ranks *filename*'s best tag; None if it does
    not install it."""
    match = _WHEEL.fullmatch(filename)
    if match is None:
        return None
    supported = _supported_tags()
    ranks = [
        supported[(python, abi, platform)]
        for python in match["python"].lower().split(".")
        for abi in match["abi"].lower().split(".")
        for platform in match["platform"].lower().split(".")
        if (python, abi, platform) in supported
    ]
    return min(ranks) if ranks else None


def _empty_manifest() -> dict:
    return {"version": 1, "wheels": {}}


def _read_manifest(cache: str) -> dict:
    path = os.path.join(cache, MANIFEST)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError:
        return _empty_manifest()
    except (OSError, ValueError) as error:
        raise InstallerError(
            f"the package cache's manifest, {path}, cannot be read "
            f"({error}). Nothing was installed."
        ) from None
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or not isinstance(manifest.get("wheels"), dict)
    ):
        raise InstallerError(
            f"the package cache's manifest, {path}, is not one this "
            f"version of g.Pype writes. Nothing was installed."
        )
    return manifest


def _write_manifest(cache: str, manifest: dict) -> None:
    handle, temporary = tempfile.mkstemp(
        prefix=".manifest-", suffix=".json", dir=cache
    )
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    os.replace(temporary, os.path.join(cache, MANIFEST))


def _valid_entry(file, entry) -> Pin:
    """A manifest entry's pin, its wheel file name and hash checked.

    The manifest lives on a volume others can write, so its entries are
    read as untrusted input.

    Args:
        file: The entry's key, the wheel's file name in the cache.
        entry: ``{"requirement", "sha256", "installed"}``.

    Raises:
        ValueError: Saying what is wrong with the entry.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"the manifest entry {_shown(file)} is no mapping")
    pin = _pin_of(entry.get("requirement"))
    if pin is None:
        raise ValueError(
            f"the manifest entry {_shown(file)} names no exact pin"
        )
    if (
        not isinstance(file, str)
        or os.path.basename(file) != file
        or "/" in file
        or "\\" in file
        or not _wheel_of(file, pin)
    ):
        raise ValueError(
            f"the manifest entry {_shown(file)} is no wheel file of {pin}"
        )
    sha256 = entry.get("sha256")
    if not isinstance(sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", sha256
    ):
        raise ValueError(f"the manifest records no sha256 for {file}")
    return pin


def _cached(cache: str, manifest: dict, pin: Pin) -> Optional[tuple]:
    """The cache's wheel of *pin* for this interpreter, or None.

    Of the recorded wheels of *pin*, the one whose tags this interpreter
    ranks best. One it cannot install is passed over, so the right one is
    downloaded; one whose file is gone is dropped from *manifest*.

    Returns:
        ``(file, sha256)``, or None to download it.
    """
    best = None
    for file, entry in list(manifest["wheels"].items()):
        try:
            recorded = _valid_entry(file, entry)
        except ValueError:
            continue
        if recorded.key != pin.key or not satisfies(
            pin.version, recorded.version
        ):
            continue
        rank = _rank(file)
        if rank is None:
            continue
        if not os.path.isfile(os.path.join(cache, file)):
            del manifest["wheels"][file]
            continue
        if best is None or rank < best[0]:
            best = (rank, file, entry["sha256"])
    return None if best is None else best[1:]


_thread_lock = threading.Lock()


def _lock(handle) -> None:
    deadline = time.monotonic() + LOCK_TIMEOUT_S
    while True:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if time.monotonic() > deadline:
                raise InstallerError(
                    f"another install held the package cache for longer "
                    f"than {LOCK_TIMEOUT_S:.0f} s. Nothing was installed."
                ) from None
            time.sleep(0.1)


def _unlock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _cache_errors(cache: str):
    """Report a file system failure as an InstallerError."""
    try:
        yield
    except OSError as error:
        raise InstallerError(
            f"the package cache, {cache}, or a private directory beside "
            f"pip could not be used: {error}"
        ) from None


@contextlib.contextmanager
def _cache_locked(cache: str):
    """Hold the cache: one install at a time, in this process and others."""
    with _thread_lock:
        os.makedirs(cache, exist_ok=True)
        with open(os.path.join(cache, LOCK_FILE), "a+b") as handle:
            _lock(handle)
            try:
                yield
            finally:
                _unlock(handle)


@contextlib.contextmanager
def _private_directory():
    """A fresh directory outside the cache, for one operation's pip calls.

    It holds :data:`WHEELS`, exactly the wheels pip may read, and
    :data:`WORKDIR`, pip's empty working directory; and it is removed
    afterwards. It is in TEMP, which must therefore hold a copy of every
    wheel the operation installs, and while a restore goes entry by
    entry one more copy of the largest.
    """
    directory = tempfile.mkdtemp(prefix="gpype-pip-")
    try:
        os.mkdir(os.path.join(directory, WHEELS))
        os.mkdir(os.path.join(directory, WORKDIR))
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _workdir(private: str) -> str:
    return os.path.join(private, WORKDIR)


def _take(source_dir: str, file: str, sha256: str, target_dir: str, pin):
    """Copy a wheel into *target_dir*, and check the copy's hash.

    The copy is what pip reads, so the bytes checked are the bytes
    installed. A copy that fails the check is removed again, so the
    directory holds only verified wheels.

    Raises:
        InstallerError: If the wheel cannot be read, or its copy does not
            match the sha256 recorded when it was downloaded.
    """
    source = os.path.join(source_dir, file)
    target = os.path.join(target_dir, file)
    try:
        shutil.copyfile(source, target)
        matches = _sha256(target) == sha256
    except OSError as error:
        failure = (
            f"the package cache's wheel of {pin}, {source}, cannot be read "
            f"({error}), so it is not installed."
        )
    else:
        if matches:
            return
        failure = (
            f"the package cache's wheel of {pin}, {source}, does not match "
            f"the sha256 recorded when it was downloaded, so it was "
            f"replaced since, and it is not installed. Remove it and its "
            f"entry in {MANIFEST} to download it again."
        )
    with contextlib.suppress(OSError):
        os.remove(target)
    raise InstallerError(failure)


def _store(cache: str, directory: str, file: str) -> None:
    """Put a downloaded wheel into the cache: a copy beside it, then an
    atomic rename, since the cache may be on another file system."""
    handle, incoming = tempfile.mkstemp(
        prefix=".incoming-", suffix=".part", dir=cache
    )
    os.close(handle)
    try:
        shutil.copyfile(os.path.join(directory, file), incoming)
        os.replace(incoming, os.path.join(cache, file))
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(incoming)
        raise


# ----------------------------------------------
# download, plan, install


def _download(run: Runner, index: str, private: str, pin: Pin) -> tuple:
    """Download one pin's wheel into *private*'s wheels, and hash it.

    Into a fresh directory of its own first, so nothing else is taken for
    what pip downloaded. The cache sees it only once the plan has passed.

    Returns:
        ``(file, sha256)``.

    Raises:
        InstallerError: If pip fails or does not produce one wheel of it.
    """
    staging = tempfile.mkdtemp(prefix="download-", dir=private)
    try:
        result = _call(
            run,
            _pip(
                "download",
                *PIP_OPTIONS,
                "--no-deps",
                "--only-binary=:all:",
                "--index-url",
                index,
                "--dest",
                staging,
                str(pin),
            ),
            f"downloading {pin}",
            private,
        )
        if result.returncode != 0:
            raise InstallerError(
                f"pip could not download {pin} from {redact_url(index)}. "
                f"Nothing was installed. pip said:\n{_output(result)}"
            )
        names = sorted(os.listdir(staging))
        if len(names) != 1 or not _wheel_of(names[0], pin):
            raise InstallerError(
                f"downloading {pin} gave {names or 'nothing'}, not one "
                f"wheel of it. Nothing was installed."
            )
        target = os.path.join(private, WHEELS, names[0])
        os.replace(os.path.join(staging, names[0]), target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return names[0], _sha256(target)


_UNSATISFIED = re.compile(
    r"(?:satisfies the requirement|No matching distribution found for) "
    r"(?P<requirement>[^\s(]+)(?: \(from (?P<parent>[^)]*)\))?"
)


def _requires(path: str) -> list:
    """The names a wheel requires unconditionally, from its METADATA."""
    names = []
    try:
        with zipfile.ZipFile(path) as wheel:
            member = next(
                (
                    name
                    for name in wheel.namelist()
                    if name.count("/") == 1
                    and name.endswith(".dist-info/METADATA")
                ),
                None,
            )
            if member is None:
                return names
            text = wheel.read(member).decode("utf-8", "replace")
    except (OSError, zipfile.BadZipFile):
        return names
    for line in text.split("\n\n", 1)[0].splitlines():
        if not line.lower().startswith("requires-dist:"):
            continue
        requirement = line.split(":", 1)[1].strip()
        if ";" in requirement:
            # A marker this module cannot evaluate; pip's plan decides.
            continue
        match = re.match(_NAME, requirement)
        if match:
            names.append(match.group(0))
    return names


def _unsatisfied(pins: list, paths: dict, output: str) -> tuple:
    """What the pinned wheels need that this environment cannot give.

    pip's resolver stops at the first requirement it cannot satisfy, so
    its output names one; each wheel's own unconditional requirements
    name the rest of those nothing provides.

    Returns:
        ``(conflicts, unpinned)``: ``(requirement, installed version,
        required by)`` for each distribution installed at a version the
        requirement excludes, and each name nothing pins or provides.
    """
    pinned = {pin.key for pin in pins}
    conflicts, unpinned = {}, {}
    for match in _UNSATISFIED.finditer(output):
        requirement = match["requirement"]
        name = re.match(_NAME, requirement)
        if not name or normalize(name.group(0)) in pinned:
            continue
        key = normalize(name.group(0))
        version = installed_version(name.group(0))
        if version is None:
            unpinned.setdefault(key, requirement)
        else:
            conflicts.setdefault(key, (requirement, version, match["parent"]))
    for pin in pins:
        for name in _requires(paths[pin]):
            key = normalize(name)
            if (
                key not in pinned
                and key not in conflicts
                and installed_version(name) is None
            ):
                unpinned.setdefault(key, name)
    return (
        [conflicts[key] for key in sorted(conflicts)],
        [unpinned[key] for key in sorted(unpinned)],
    )


def _refuse_unplanned(pins: list, paths: dict, output: str) -> None:
    """Refuse a plan pip could not make, naming why if its output can.

    Raises:
        RequirementConflictError: If a wheel needs a version other than
            the one installed here.
        RequirementError: If a wheel needs what nothing pins or provides.
        InstallerError: Otherwise.
    """
    conflicts, unpinned = _unsatisfied(pins, paths, output)
    closure = (
        "The pins are the closure: a distribution installed here "
        "satisfies a requirement as installed, and anything else a pinned "
        "wheel needs must be pinned too."
    )
    if conflicts:
        named = ", ".join(
            (f"{parent} needs " if parent else "")
            + f"{requirement} (installed here: {version})"
            for requirement, version, parent in conflicts
        )
        also = (
            f" They also need, unpinned: {', '.join(unpinned)}."
            if unpinned
            else ""
        )
        raise RequirementConflictError(
            f"the pinned wheels need {len(conflicts)} distribution(s) at a "
            f"version other than the one installed here: {named}.{also} "
            f"One environment serves every document this deployment loads, "
            f"so an installed distribution is never replaced, and a pin of "
            f"it would be refused as well. Nothing was installed. pip "
            f"said:\n{output}",
            [requirement for requirement, _, _ in conflicts],
        )
    if unpinned:
        raise RequirementError(
            f"the pinned wheels need {len(unpinned)} distribution(s) this "
            f"document does not pin and this environment does not have: "
            f"{', '.join(unpinned)}. {closure} Nothing was installed. pip "
            f"said:\n{output}",
            unpinned,
        )
    raise InstallerError(
        f"pip could not plan installing {', '.join(map(str, pins))} from "
        f"the package cache's wheels. Nothing was installed. pip "
        f"said:\n{output}"
    )


def _read_plan(path: str) -> list:
    """``[(name, version, sha256 or None)]`` from pip's install report.

    Raises:
        InstallerError: If the report cannot be read.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        plan = []
        for item in report.get("install") or []:
            metadata = item["metadata"]
            archive = (item.get("download_info") or {}).get(
                "archive_info"
            ) or {}
            sha256 = (archive.get("hashes") or {}).get("sha256")
            if sha256 is None and str(archive.get("hash", "")).startswith(
                "sha256="
            ):
                sha256 = archive["hash"][len("sha256=") :]
            plan.append(
                (str(metadata["name"]), str(metadata["version"]), sha256)
            )
        return plan
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise InstallerError(
            f"pip's install plan could not be read ({e}). Nothing was "
            f"installed."
        ) from None


def _check_plan(plan: list, pins: list, hashes: dict) -> None:
    """Refuse a plan other than exactly *pins*, from their verified wheels.

    Raises:
        RequirementError: If the plan installs anything else: an
            unpinned requirement, or another version of an installed
            distribution.
        InstallerError: If it leaves out a pin, or offers a wheel other
            than the one recorded.
    """
    expected = {pin.key: pin for pin in pins}
    extra = []
    for name, version, sha256 in plan:
        pin = expected.pop(normalize(name), None)
        if pin is None or not satisfies(pin.version, version):
            installed = installed_version(name)
            extra.append(
                f"{name} {version}"
                + (
                    f" (would replace the installed {installed})"
                    if installed
                    else " (not pinned)"
                )
            )
        elif sha256 is not None and sha256 != hashes[pin]:
            raise InstallerError(
                f"pip's plan offers a wheel of {pin} whose sha256 is not "
                f"the one recorded at its download. Nothing was installed."
            )
    if extra:
        raise RequirementError(
            f"installing this document's pins would also install "
            f"{', '.join(extra)}. The pins are the closure: a distribution "
            f"installed here satisfies a requirement as installed, and "
            f"anything else a pinned wheel needs must be pinned too. "
            f"Nothing was installed.",
            extra,
        )
    if expected:
        missing = ", ".join(str(pin) for pin in expected.values())
        raise InstallerError(
            f"pip's plan leaves out {missing}, which this process does not "
            f"find installed. Nothing was installed."
        )


def _own(directory: str, suffix: str, pin: Pin) -> bool:
    """Whether a wheel's top-level *directory* is its own, ending in
    *suffix*, such as its ``.dist-info``."""
    if not directory.lower().endswith(suffix):
        return False
    stem = directory[: -len(suffix)].rpartition("-")[0]
    return normalize(stem) == pin.key


#: Where pip puts what a wheel's ``.data`` directory holds. ``site`` is
#: site-packages, which takes the wheel's root too.
_DATA_SCHEMES = {
    "purelib": "site",
    "platlib": "site",
    "scripts": "scripts",
    "data": "data",
    "headers": "headers",
}

#: What a module's file name may end with here, longest first.
_MODULE_SUFFIXES = tuple(
    sorted(set(importlib.machinery.all_suffixes()), key=len, reverse=True)
)


def _install_scheme(pin: Pin) -> dict:
    """Where pip installs a wheel's ``.data`` scripts, data and headers.

    From sysconfig, as pip maps them: headers go to
    ``include/site/pythonX.Y/<name>`` in a virtual environment, as pip
    has always put them there, and under sysconfig's include otherwise.

    Returns:
        ``{"scripts": dir, "data": dir, "headers": dir}``.
    """
    paths = sysconfig.get_paths()
    include = paths["include"]
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        include = os.path.join(
            sys.prefix, "include", "site", "python%d.%d" % sys.version_info[:2]
        )
    return {
        "scripts": os.path.abspath(paths["scripts"]),
        "data": os.path.abspath(paths["data"]),
        "headers": os.path.join(os.path.abspath(include), pin.name),
    }


def _normalized(name: str) -> Optional[str]:
    """A wheel member's path as pip resolves it, ``a/b`` with ``..``
    folded; None where pip refuses it, outside its directory."""
    if os.name == "nt":
        name = name.replace("\\", "/")
    name = posixpath.normpath(name)
    first = name.split("/", 1)[0]
    if first in ("", ".", "..") or (os.name == "nt" and ":" in first):
        return None
    return name


def _launchers(text: str) -> list:
    """The scripts pip generates for an ``entry_points.txt``'s console and
    GUI entry points, by file name. Only ``#`` starts a comment, as
    ``importlib.metadata`` reads the file."""
    names, section = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
        elif section in ("console_scripts", "gui_scripts") and "=" in line:
            name = line.split("=", 1)[0].strip()
            if name:
                names.append(f"{name}.exe" if os.name == "nt" else name)
    return names


def _targets(wheel: zipfile.ZipFile, pin: Pin) -> dict:
    """What installing *wheel* writes, as a tree per place pip puts it.

    ``{key: tree}``: *key* is ``site`` for site-packages, which takes the
    wheel's root, less its own ``.dist-info``, and any ``.data``
    directory's purelib and platlib, as pip takes them; else
    ``scripts``, ``data`` or ``headers``. A tree maps a directory's name
    to its own tree, and a file's to its member in the wheel, or to None
    for a script pip generates from an entry point. A path pip refuses,
    outside its directory or under a scheme it does not know, is left
    out: pip refuses the wheel then.

    Raises:
        InstallerError: If the wheel's entry points cannot be read.
    """
    trees: dict = {}

    def add(key: str, target: str, member: Optional[str]) -> None:
        parts = target.split("/")
        node = trees.setdefault(key, {})
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        if not isinstance(node.get(parts[-1]), dict):
            node[parts[-1]] = member

    failure = None
    for name in wheel.namelist():
        if name.endswith("/"):
            continue
        first, _, rest = name.partition("/")
        if rest == "entry_points.txt" and _own(first, ".dist-info", pin):
            try:
                text = wheel.read(name).decode("utf-8", "replace")
            except Exception as error:  # noqa: BLE001 - any unreadable zip
                failure = f"its entry points cannot be read ({error})"
            else:
                for launcher in _launchers(text):
                    target = _normalized(launcher)
                    if target is not None:
                        add("scripts", target, None)
        normed = _normalized(name)
        if normed is None:
            continue
        if first.endswith(".data"):
            # As pip reads it: any .data directory, its scheme the second
            # part of the folded path.
            parts = normed.split("/", 2)
            key = _DATA_SCHEMES.get(parts[1]) if len(parts) == 3 else None
            if key is not None:
                add(key, parts[2], name)
        elif not _own(normed.split("/", 1)[0], ".dist-info", pin):
            add("site", normed, name)
    if failure:
        raise InstallerError(
            f"the wheel of {pin} cannot be read: {failure}. Nothing was "
            f"installed."
        )
    return trees


def _same_bytes(wheel: zipfile.ZipFile, member: str, path: str) -> bool:
    """Whether the file at *path* holds exactly *member*'s bytes."""
    try:
        info = wheel.getinfo(member)
        if not os.path.isfile(path) or os.path.getsize(path) != info.file_size:
            return False
        digest = hashlib.sha256()
        with wheel.open(info) as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest() == _sha256(path)
    except Exception:  # noqa: BLE001 - an unreadable file is no match
        return False


def _import_name(name: str, is_dir: bool) -> Optional[str]:
    """The name *name* is imported by; None for a file that is no module."""
    if is_dir:
        return name
    for suffix in _MODULE_SUFFIXES:
        if len(name) > len(suffix) and name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _shadowed(directory: str, name: str, is_dir: bool) -> Optional[str]:
    """What in *directory* is *name*'s import name in another form.

    A package ``x/``, ``x.py`` and ``x.<suffix>`` are one import name, and
    the import system takes one of them, so writing one beside another
    changes what ``import x`` gives.
    """
    stem = _import_name(name, is_dir)
    if stem is None:
        return None
    for form in (stem, *(stem + suffix for suffix in _MODULE_SUFFIXES)):
        if form != name and os.path.lexists(os.path.join(directory, form)):
            return form
    return None


def _shared(directory: str, tree: dict, same) -> bool:
    """Whether a directory both the wheel and this environment have is one
    to look into: a namespace package, with an ``__init__`` on neither
    side, or a package whose ``__init__`` files are the same bytes on
    both, as a distribution split in several shares one."""
    ours = {
        name: member
        for name, member in tree.items()
        if not isinstance(member, dict)
        and _import_name(name, False) == "__init__"
    }
    theirs = {
        form
        for form in ("__init__" + suffix for suffix in _MODULE_SUFFIXES)
        if os.path.lexists(os.path.join(directory, form))
    }
    if {os.path.normcase(name) for name in ours} != {
        os.path.normcase(name) for name in theirs
    }:
        return False
    return all(
        member is not None and same(member, os.path.join(directory, name))
        for name, member in ours.items()
    )


def _walk(directory: str, tree: dict, same, shadows: bool, prefix=""):
    """``(ours, theirs)`` for each place under *directory* where writing
    *tree* would change what is there, as ``a/b`` paths.

    A path that exists is such a place unless it is a file holding the
    same bytes, or a directory :func:`_shared` says to look into. With
    *shadows*, so is another form of one of the tree's import names.
    """
    found = []
    for name in sorted(tree):
        child = tree[name]
        is_dir = isinstance(child, dict)
        ours = prefix + name
        form = _shadowed(directory, name, is_dir) if shadows else None
        if form is not None:
            found.append((ours, prefix + form))
            continue
        path = os.path.join(directory, name)
        if not os.path.lexists(path):
            continue
        if is_dir and os.path.isdir(path) and _shared(path, child, same):
            found += _walk(path, child, same, shadows, ours + "/")
        elif is_dir or child is None or not same(child, path):
            found.append((ours, ours))
    return found


def _owners(paths: set) -> dict:
    """The installed distribution that recorded each of *paths*, or a file
    under it.

    From each distribution's own record of its files; a path none
    records, such as what an interrupted install left or the
    interpreter's own ``pyvenv.cfg``, has no owner.
    """
    if not paths:
        return {}
    # Folded, several spellings of one path share a key; each is answered.
    wanted: dict = {}
    for path in paths:
        wanted.setdefault(_fold(os.path.abspath(path)), []).append(path)
    under = tuple(key + os.sep for key in wanted)
    owners = {}
    for distribution in importlib.metadata.distributions():
        try:
            files = distribution.files or []
            owner = f"{distribution.metadata['Name']} {distribution.version}"
            base = os.path.abspath(str(distribution.locate_file("")))
        except Exception:  # noqa: BLE001 - a broken record owns nothing
            continue
        for file in files:
            located = _fold(os.path.join(base, str(file)))
            if located not in wanted and not located.startswith(under):
                continue
            for key, spellings in wanted.items():
                if located == key or located.startswith(key + os.sep):
                    for path in spellings:
                        owners.setdefault(path, owner)
        if len(owners) == len(paths):
            break
    return owners


def _written(key: str, ours: str, theirs: str) -> str:
    """What a pin writes that collides, for the message."""
    where = "" if key == "site" else f" in the {key} directory"
    if ours == theirs:
        return f"writes {ours}{where}"
    return f"writes {ours}{where}, shadowing {theirs}"


def _in_site_packages(path: str, sites: list) -> bool:
    folded = _fold(os.path.abspath(path))
    return any(
        folded.startswith(_fold(site).rstrip(os.sep) + os.sep)
        for site in sites
    )


def _check_collisions(pins: list, paths: dict) -> None:
    """Refuse a wheel that would change another distribution's files, or
    the environment's own.

    One environment serves every document (D-CORE-87): a pin that ships
    another distribution's modules, as opencv-python-headless does
    opencv-python's ``cv2``, would replace code under a running session.
    Every place pip would write (:func:`_targets`) is compared with what
    the installed distributions recorded. In site-packages, a path none
    records is what an interrupted install left, and is overwritten;
    elsewhere it is the environment's own, such as ``pyvenv.cfg`` or
    ``Scripts/pythonw.exe``, and is refused (D-CORE-119).

    Raises:
        RequirementConflictError: Naming each pin, what it writes and
            the distribution that installed it, or that none did.
        InstallerError: If a wheel cannot be read.
    """
    sites = _site_packages()
    candidates = []
    for pin in pins:
        try:
            wheel = zipfile.ZipFile(paths[pin])
        except (OSError, zipfile.BadZipFile) as error:
            failure = f"the wheel of {pin} cannot be read ({error})."
        else:
            failure = None
        if failure:
            raise InstallerError(f"{failure} Nothing was installed.")
        with wheel:
            roots = {"site": sites}
            roots.update(
                (key, [root]) for key, root in _install_scheme(pin).items()
            )
            for key, tree in _targets(wheel, pin).items():
                shadows = key in ("site", "data")
                same = functools.partial(_same_bytes, wheel)
                for root in roots[key]:
                    for ours, theirs in _walk(root, tree, same, shadows):
                        place = os.path.join(root, *theirs.split("/"))
                        candidates.append((pin, key, ours, theirs, place))
    owners = _owners({candidate[-1] for candidate in candidates})
    found, seen = [], set()
    for pin, key, ours, theirs, place in candidates:
        owner = owners.get(place)
        if owner is None and _in_site_packages(place, sites):
            continue
        if (pin, key, ours, theirs) not in seen:
            seen.add((pin, key, ours, theirs))
            found.append((pin, _written(key, ours, theirs), owner))
    if found:
        named = "; ".join(
            f"{pin} {what}, which "
            + (f"{owner} installed" if owner else "no distribution records")
            for pin, what, owner in found
        )
        offenders = []
        for pin, _, _ in found:
            if str(pin) not in offenders:
                offenders.append(str(pin))
        raise RequirementConflictError(
            f"installing this document's pins would overwrite or shadow "
            f"files another distribution installed here, or this "
            f"environment's own: {named}. One environment serves every "
            f"document this deployment loads, so a pin never replaces "
            f"another distribution's code, nor a file outside "
            f"site-packages that no distribution records. Nothing was "
            f"installed.",
            offenders,
        )


def _install_options(directory: str) -> tuple:
    return (
        "install",
        *PIP_OPTIONS,
        "--no-user",
        "--no-index",
        "--find-links",
        directory,
        "--only-binary=:all:",
    )


def _plan(run: Runner, private: str, directory: str, pins, wheels) -> None:
    """Plan installing *pins* from *directory* alone, and refuse any plan
    but exactly them, or one that overwrites another distribution.

    The pins go to pip as arguments, not as the hashed file: a ``--hash``
    line turns on pip's hash-checking mode, which fails on an unpinned
    dependency instead of reporting it in the plan.

    Args:
        directory: Holds exactly the verified wheels.
        wheels: ``{pin: (file, sha256)}``.
    """
    paths = {pin: os.path.join(directory, wheels[pin][0]) for pin in pins}
    work = tempfile.mkdtemp(prefix="plan-", dir=private)
    try:
        report = os.path.join(work, "plan.json")
        result = _call(
            run,
            _pip(
                *_install_options(directory),
                "--dry-run",
                "--report",
                report,
                *map(str, pins),
            ),
            "planning the install",
            private,
        )
        if result.returncode != 0:
            _refuse_unplanned(pins, paths, _output(result))
        plan = _read_plan(report)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    _check_plan(plan, pins, {pin: wheels[pin][1] for pin in pins})
    _check_collisions(pins, paths)


def _install(run: Runner, private: str, directory: str, pins, wheels):
    """Install *pins* from *directory*, pip checking each hash again."""
    work = tempfile.mkdtemp(prefix="install-", dir=private)
    try:
        hashed = os.path.join(work, "requirements.txt")
        with open(hashed, "w", encoding="utf-8") as handle:
            for pin in pins:
                handle.write(f"{pin} --hash=sha256:{wheels[pin][1]}\n")
        result = _call(
            run,
            _pip(
                *_install_options(directory), "--require-hashes", "-r", hashed
            ),
            "installing",
            private,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    importlib.invalidate_caches()
    if result.returncode != 0:
        raise InstallerError(
            f"pip could not install {', '.join(map(str, pins))} from "
            f"the package cache's wheels. pip said:\n{_output(result)}"
        )


def _mark_installed(manifest: dict, files: set) -> None:
    """Record exactly the wheels *files* as installed, and no other
    version of each; another wheel of the same version is left as is."""
    used = {}
    for file in files:
        match = _WHEEL.fullmatch(file)
        used[normalize(match["name"])] = match["version"]
    for file, entry in manifest["wheels"].items():
        if not isinstance(entry, dict):
            continue
        if file in files:
            entry["installed"] = True
            continue
        match = _WHEEL.fullmatch(file) if isinstance(file, str) else None
        key = normalize(match["name"]) if match else None
        if key in used and _version_key(used[key]) != _version_key(
            match["version"]
        ):
            entry["installed"] = False


def install(requirements, runner: Optional[Runner] = None) -> dict:
    """Install every pin this environment lacks, from the configured index.

    Conflicts are refused first, and a site-packages this process cannot
    write. Then, under the cache's lock, each absent pin's wheel is copied
    from the cache or downloaded (``--no-deps``, wheels only) into a
    private directory and hashed; pip plans an install of the absent pins
    from that directory alone, and the install runs only if the plan is
    exactly them and overwrites or shadows no other distribution's files.
    A download joins the cache, and the manifest, only once the plan has
    passed.

    Args:
        requirements: A document's pins, as written.
        runner: Runs one pip command; the module's runner when None.

    Returns:
        ``{"installed": [...], "present": [...]}``: the pins installed now,
        and those already installed.

    Raises:
        RequirementError: If a pin is not exact, or a pinned wheel needs
            a distribution nothing pins or provides.
        RequirementConflictError: If a pin differs from what is
            installed, a wheel needs another version of an installed
            distribution, or it would overwrite or shadow one's files.
        InstallerError: If the deployment's index, the interpreter, its
            site-packages, pip or the cache cannot install, or a cached
            wheel was replaced.
    """
    pins = parse_pins(requirements)
    check_conflicts(pins)
    if not _absent(pins):
        return {"installed": [], "present": [str(pin) for pin in pins]}
    index = _index()
    _require_pip()
    _require_site_packages()
    run = runner or _runner
    cache = _config().cache
    with _cache_errors(cache), _cache_locked(cache):
        # Another install may have finished while this one waited.
        check_conflicts(pins)
        absent = _absent(pins)
        present = [str(pin) for pin in pins if pin not in absent]
        if absent:
            manifest = _read_manifest(cache)
            with _private_directory() as private:
                directory = os.path.join(private, WHEELS)
                wheels, downloaded = {}, []
                for pin in absent:
                    found = _cached(cache, manifest, pin)
                    if found is None:
                        wheels[pin] = _download(run, index, private, pin)
                        downloaded.append(pin)
                    else:
                        _take(cache, *found, directory, pin)
                        wheels[pin] = found
                _plan(run, private, directory, absent, wheels)
                for pin in downloaded:
                    file, sha256 = wheels[pin]
                    _store(cache, directory, file)
                    manifest["wheels"][file] = {
                        "requirement": str(pin),
                        "sha256": sha256,
                        "installed": False,
                    }
                if downloaded:
                    _write_manifest(cache, manifest)
                _install(run, private, directory, absent, wheels)
            _mark_installed(manifest, {wheels[pin][0] for pin in absent})
            _write_manifest(cache, manifest)
    return {"installed": [str(pin) for pin in absent], "present": present}


def _restorable(cache: str, manifest: dict, report: dict, skip) -> list:
    """The wheels a restore installs, one per recorded pin.

    Of each pin's wheels recorded as installed, the one this interpreter
    ranks best. Each entry passed over is reported: invalid, installed at
    another version, a second version, no wheel this interpreter
    installs, or a wheel gone from the cache.

    Returns:
        ``[(pin, file, sha256)]``.
    """
    groups = {}
    for file in sorted(manifest["wheels"], key=str):
        entry = manifest["wheels"][file]
        if not isinstance(entry, dict) or entry.get("installed") is not True:
            continue
        try:
            pin = _valid_entry(file, entry)
        except ValueError as reason:
            skip(str(entry.get("requirement"))[:120], str(reason))
            continue
        group = (pin.key, _version_key(pin.version))
        groups.setdefault(group, []).append((file, pin, entry["sha256"]))
    chosen, names = [], set()
    for (key, _), members in groups.items():
        pin = members[0][1]
        version = installed_version(pin.name)
        if version is not None:
            if satisfies(pin.version, version):
                report["present"].append(str(pin))
            else:
                skip(str(pin), f"{version} is installed here")
            continue
        if key in names:
            skip(str(pin), "another version of it is recorded too")
            continue
        ranked = []
        for file, _, sha256 in members:
            rank = _rank(file)
            if rank is not None:
                ranked.append((rank, file, sha256))
        if not ranked:
            skip(
                str(pin),
                f"no wheel of it in the package cache is one this "
                f"interpreter installs: "
                f"{', '.join(file for file, _, _ in members)}",
            )
            continue
        _, file, sha256 = min(ranked)
        if not os.path.isfile(os.path.join(cache, file)):
            skip(str(pin), f"its wheel, {file}, is not in the package cache")
            continue
        names.add(key)
        chosen.append((pin, file, sha256))
    return chosen


def _attempt(run: Runner, private: str, directory: str, pins, wheels):
    """Plan and install *pins*; the refusal, or None once installed.

    A pip call that does not finish is a refusal like any other.

    Raises:
        InstallerError: Only if pip cannot be started.
    """
    try:
        _plan(run, private, directory, pins, wheels)
        _install(run, private, directory, pins, wheels)
    except _PipCannotRun:
        raise
    except (RequirementError, RequirementConflictError, InstallerError) as e:
        return e
    return None


def _attempt_alone(run: Runner, private: str, directory: str, pin, wheels):
    """Plan and install *pin* from a directory holding only its wheel,
    removed right afterwards, so TEMP holds one extra copy at a time;
    the refusal, or None once installed."""
    single = tempfile.mkdtemp(prefix="entry-", dir=private)
    try:
        try:
            _take(directory, *wheels[pin], single, pin)
        except InstallerError as error:
            return error
        return _attempt(run, private, single, [pin], wheels)
    finally:
        shutil.rmtree(single, ignore_errors=True)


def _installed_as(pin: Pin) -> bool:
    version = installed_version(pin.name)
    return version is not None and satisfies(pin.version, version)


def _restore_wheels(run: Runner, private: str, wheels: dict, skip) -> list:
    """Install *wheels*: at once if pip plans them so, else entry by entry.

    Entry by entry, the pass repeats while any entry installs, so one
    that needs another recorded pin follows it. One whose own pip call
    did not finish is not tried again. Each still refused is skipped
    with its reason.

    Returns:
        The pins installed.

    Raises:
        InstallerError: If pip cannot be started.
    """
    pins = list(wheels)
    directory = os.path.join(private, WHEELS)
    failure = _attempt(run, private, directory, pins, wheels)
    if failure is None:
        return [str(pin) for pin in pins]
    installed, reasons, remaining = [], {}, list(pins)
    progress = True
    while remaining and progress:
        progress = False
        for pin in list(remaining):
            if not _installed_as(pin):
                reasons[pin] = _attempt_alone(
                    run, private, directory, pin, wheels
                )
                if isinstance(reasons[pin], _PipTimedOut):
                    remaining.remove(pin)
                if reasons[pin] is not None:
                    continue
            installed.append(str(pin))
            remaining.remove(pin)
            progress = True
    for pin in pins:
        if str(pin) not in installed:
            skip(str(pin), str(reasons.get(pin) or failure))
    return installed


def restore(runner: Optional[Runner] = None) -> dict:
    """Reinstall the pins the cache records as installed, with no network.

    For the process that starts a deployment (D-CORE-88). Entry by entry:
    one installed at another version now, a second version of one, one
    whose wheel this interpreter does not install, is missing or was
    replaced, one pip refuses or that would overwrite another
    distribution's files, and one whose own pip call does not finish
    within :data:`PIP_TIMEOUT_S` is skipped and reported, and the others
    are installed. Does nothing without the opt-in.

    Args:
        runner: Runs one pip command; the module's runner when None.

    Returns:
        ``{"enabled": bool, "installed": [...], "present": [...],
        "skipped": [{"requirement": str, "reason": str}]}``, JSON-safe.

    Raises:
        InstallerError: Before anything is installed, if the cache's
            manifest cannot be read, another install holds the cache for
            longer than :data:`LOCK_TIMEOUT_S`, this interpreter's wheel
            tags cannot be read, or the interpreter, its site-packages or
            pip cannot install. At any point, possibly after some entries
            were installed, which the next restore reports as present, if
            pip cannot be started or the cache or a private directory
            beside pip cannot be used.
    """
    report = {
        "enabled": enabled(),
        "installed": [],
        "present": [],
        "skipped": [],
    }
    cache = _config().cache
    if not report["enabled"] or not os.path.isfile(
        os.path.join(cache, MANIFEST)
    ):
        return report

    def skip(requirement, reason) -> None:
        report["skipped"].append(
            {"requirement": redact(requirement), "reason": redact(reason)}
        )

    with _cache_errors(cache), _cache_locked(cache):
        manifest = _read_manifest(cache)
        chosen = _restorable(cache, manifest, report, skip)
        if chosen:
            _require_pip()
            _require_site_packages()
            with _private_directory() as private:
                directory = os.path.join(private, WHEELS)
                wheels = {}
                for pin, file, sha256 in chosen:
                    try:
                        _take(cache, file, sha256, directory, pin)
                    except InstallerError as error:
                        skip(str(pin), str(error))
                        continue
                    wheels[pin] = (file, sha256)
                if wheels:
                    report["installed"] = _restore_wheels(
                        runner or _runner, private, wheels, skip
                    )
    importlib.invalidate_caches()
    return report


# ----------------------------------------------
# what Pipeline.check asks


def for_document(
    requirements, install_absent: bool, runner: Optional[Runner] = None
) -> Optional[dict]:
    """The pins' step of ``Pipeline.check``, before the requirement check.

    Nothing without the opt-in or without pins (D-CORE-111). With both:
    the pins are parsed and checked for conflicts, and with
    *install_absent* every absent pin is installed (D-CORE-114).

    Args:
        requirements: The document's ``requirements`` value.
        install_absent: Whether to install what is absent.
        runner: Runs one pip command; the module's runner when None.

    Returns:
        None, or ``{"pins": [...], "installed": [...], "absent": [...]}``:
        every pin, those installed now, and those still absent.

    Raises:
        RequirementError, RequirementConflictError, InstallerError: As
            :func:`install` raises them.
    """
    if not enabled() or not requirements:
        return None
    pins = parse_pins(requirements)
    check_conflicts(pins)
    installed = []
    if install_absent:
        installed = install([str(pin) for pin in pins], runner)["installed"]
    return {
        "pins": [str(pin) for pin in pins],
        "installed": installed,
        "absent": [str(pin) for pin in _absent(pins)],
    }


def pins_not_installed(requirements) -> Optional[str]:
    """The sentence saying a document's pins were not installed, or None.

    Only where the deployment has not opted in and the document pins
    something: its pins install nothing then (D-CORE-111), so a refusal
    for what they would have supplied says why.

    Args:
        requirements: The document's ``requirements`` value, as written.
    """
    if enabled() or not requirements:
        return None
    if isinstance(requirements, (list, tuple)):
        shown = ", ".join(_shown(item, 80) for item in requirements[:8])
        if len(requirements) > 8:
            shown += f" and {len(requirements) - 8} more"
    else:
        shown = _shown(requirements)
    return (
        f"This document's pins, {shown}, were not installed, because this "
        f"deployment does not set {ENV_INDEX}; a deployment that sets it "
        f"installs them when the document is loaded."
    )


def missing_hint(state: Optional[dict], requirements=None) -> str:
    """The sentence a refusal for missing packages ends with.

    Args:
        state: What :func:`for_document` returned for the document.
        requirements: The document's ``requirements`` value, as written.

    Returns:
        How a pin could supply them, and whether this deployment
        installs pins.
    """
    if not enabled():
        return pins_not_installed(requirements) or (
            f"Or pin the distribution that provides each, with "
            f"serialize(requirements=[...]), where this deployment sets "
            f"{ENV_INDEX} to install a document's pins; it does not."
        )
    if state is None:
        return (
            f"This deployment installs a document's pins ({ENV_INDEX} is "
            f"set), and this document pins nothing: pin the distribution "
            f"that provides each."
        )
    installed = set(state["installed"])
    named = ", ".join(
        f"{pin} ({'installed now' if pin in installed else 'installed'})"
        for pin in state["pins"]
    )
    return (
        f"This deployment installs a document's pins ({ENV_INDEX} is "
        f"set), and none of this document's pins supplies them: {named}. "
        f"Pin the distribution that provides each."
    )
