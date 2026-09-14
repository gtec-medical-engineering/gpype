"""Global launch configuration for g.Pype applications.

This module provides the LaunchConfig singleton class for managing
startup parameters including residency mode and WebSocket endpoints.

Supports command-line argument parsing for seamless integration
with application entry points.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .._transport import (
    ClientTls,
    ServerTls,
    check_endpoint_allowed,
    endpoint_is_secure,
)
from .constants import Constants

#: Where an EDGE reads its data-plane token from when it was not passed
#: programmatically. An environment variable rather than a flag: ``argv``
#: is visible in every process listing on the machine.
ENV_DATA_TOKEN = "GPYPE_DATA_TOKEN"


def _reset_raw_state() -> None:
    """Clear the raw-recording state that outlives a configuration.

    Imported late and defensively, and both parts are deliberate.
    ``common`` reaching into ``backend`` is the inverse of this
    package's layering, so the import cannot be at module scope; and it
    is guarded, because it made reading the launch configuration depend
    on the node layer importing. On a wheel whose compiled
    ``backend.core.node`` extension does not load -- the ABI mismatch
    ``nox -s test_compiled`` exists to catch -- ``LaunchConfig.get()``
    would have failed too, so the process could no longer read its own
    configuration in order to report what went wrong.

    Skipping the reset there costs nothing: if that module cannot be
    imported, no source chain can be built either, so there are no
    stream keys, no run and no replay anchor to clear.
    """
    try:
        from ..backend.sources.base import raw
    except ImportError:  # pragma: no cover - a broken build
        return
    raw.reset_stream_keys()
    raw.RawRun.reset()
    raw.ReplayClock.reset()


@dataclass
class LaunchConfig:
    """Global launch configuration singleton.

    Stores startup parameters for the entire application. All Source/Sink
    nodes and other components read from this global configuration.

    Supports three ways to configure:
    1. Direct assignment: LaunchConfig.configure(residency=..., ...)
    2. Command-line parsing: LaunchConfig.parse_args(sys.argv[1:])
    3. Programmatic: LaunchConfig.get() then modify fields

    Usage in main():
        def main():
            LaunchConfig.parse_args()
            # ... rest of application

    Usage in nodes:
        config = LaunchConfig.get()
        if config.is_edge():
            ...
    """

    #: Residency mode: standalone, edge, or server
    residency: str = Constants.Residency.STANDALONE

    #: WebSocket endpoint URL for edge/server communication. A ``wss``
    #: scheme is what switches the data socket to TLS: both ends read
    #: the same endpoint, so they cannot disagree about whether the
    #: socket between them is encrypted.
    endpoint: Optional[str] = None

    #: Which interfaces a SERVER's broker binds. Not derivable from
    #: ``endpoint``: that host is what an edge dials, and a container
    #: advertises an address it does not bind. Inert in the other two
    #: residencies, which bind nothing.
    bind: str = Constants.Bind.ANY

    #: PEM certificate the SERVER's broker presents. Required for a
    #: ``wss`` endpoint in server residency.
    certfile: Optional[Path] = None

    #: PEM private key, if it is not inside ``certfile``.
    keyfile: Optional[Path] = None

    #: A certificate an EDGE should trust in addition to the system
    #: store. How a self-signed deployment is verified rather than
    #: waved through.
    ca_file: Optional[Path] = None

    #: Cross a network in the clear, or skip verifying a certificate.
    #: A named choice, because the samples on this socket are the
    #: recording and the alternative is silence about it.
    allow_insecure: bool = False

    #: The token this process presents when it *dials* a broker, i.e.
    #: what an EDGE sends. None means present nothing, which a broker
    #: accepts only if it requires nothing.
    data_token: Optional[str] = None

    #: The tokens this process *accepts* when it serves a broker, keyed
    #: by capability -- ``{"put": ..., "subscribe": ...}``. None means
    #: accept every connection, which is what a STANDALONE or LAN run
    #: has always done; a deployment sets it and the broker then refuses
    #: anything unauthenticated.
    #:
    #: Two capabilities rather than one, because the fan-in direction is
    #: a *single receiver slot with no sender identity*: a connection
    #: that may PUT can silently replace the stream an amplifier is
    #: feeding. Read access and write access are therefore different
    #: grants, and a viewer gets only the first.
    data_tokens: Optional[dict] = None

    #: Record every source's frames, at the point right after its core
    #: and before any Link, Oscar or Sync. A path stem: the run is a
    #: *directory* of one file per stream plus a manifest, because there
    #: is one recording per source and they only mean anything together.
    #:
    #: Here rather than on the nodes, and for the same reason residency
    #: is: it is a property of how this process was launched, not of the
    #: pipeline. One consequence is worth knowing -- it is read while the
    #: chains are built, so it must be set before the first source is
    #: constructed. A command line satisfies that by itself, since
    #: LaunchConfig parses argv on first access.
    save_as: Optional[str] = None

    #: Replay a run recorded by ``save_as``, in place of every source's
    #: own core. The run directory, or its manifest.
    load_from: Optional[str] = None

    # Singleton instance
    _instance: Optional["LaunchConfig"] = None

    @classmethod
    def get(cls) -> "LaunchConfig":
        """Get the singleton instance, creating if needed.

        On first call, parses sys.argv unless configure() was already
        called programmatically.

        Returns:
            The global LaunchConfig instance.
        """
        if cls._instance is None:
            # Reading the host's own command line is a convenience for a
            # g.Pype launcher, not a licence to fail in someone else's
            # process. A test runner or an application that happens to
            # use -r or -e for its own purposes must not be unable to
            # build a node, so an unparseable argv falls back to the
            # default rather than exiting.
            try:
                cls.parse_args()
            except SystemExit:
                # Falling back to defaults must not throw away a
                # recording the operator asked for. argparse errors on a
                # *known* flag with a bad value -- ``-r fE``, which is
                # the pytest-shaped argv this fallback exists for -- and
                # the whole command line went with it, including
                # ``--save-as``. The pipeline then ran normally with no
                # tap and nothing in g.Pype said the recording had been
                # dropped.
                #
                # So the two file parameters are recovered on their own,
                # with a parser that knows nothing else and therefore
                # cannot error on anything else.
                cls.configure(**cls._recover_file_args())
        return cls._instance

    @classmethod
    def _recover_file_args(cls, args: Optional[Sequence[str]] = None) -> dict:
        """Pull just ``--save-as``/``--load-from`` out of an argv.

        Used only on the fallback path, where the full parse has already
        failed on something else entirely. ``parse_known_args`` on a
        parser declaring nothing but these two cannot fail: every other
        token is unknown, and unknown tokens are returned rather than
        refused.

        Args:
            args: Command-line arguments. Defaults to ``sys.argv[1:]``.

        Returns:
            The keyword arguments to pass to :meth:`configure`, empty if
            neither flag is present.
        """
        # allow_abbrev off, and this is the one parser where it
        # genuinely bites: this path runs when g.Pype is embedded in a
        # host process whose argv it does not own, which is exactly
        # where a colliding prefix is likely. With abbreviation on, a
        # host's own ``--load <config>`` matched ``--load-from`` and
        # every source in the graph was replaced by a replay core
        # reading a path the operator never gave g.Pype.
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument("--save-as", dest="save_as", default=None)
        parser.add_argument("--load-from", dest="load_from", default=None)
        try:
            parsed, _ = parser.parse_known_args(
                sys.argv[1:] if args is None else args
            )
        except SystemExit:  # pragma: no cover - a flag given no value
            return {}
        if parsed.save_as and parsed.load_from:
            # Both would be refused by configure(), and this path is
            # reached from get() -- which every source chain calls while
            # it is being built. Raising there turns a command-line typo
            # into an exception from inside a node constructor, where
            # get() had never raised before. The full parse already
            # failed on something else, so the argv is being ignored
            # anyway; ignoring these two with it is consistent.
            return {}
        recovered = {}
        if parsed.save_as:
            recovered["save_as"] = parsed.save_as
        if parsed.load_from:
            recovered["load_from"] = parsed.load_from
        return recovered

    @classmethod
    def configure(
        cls,
        residency: str = Constants.Residency.STANDALONE,
        endpoint: Optional[str] = None,
        bind: str = Constants.Bind.ANY,
        certfile: Optional[Path] = None,
        keyfile: Optional[Path] = None,
        ca_file: Optional[Path] = None,
        allow_insecure: bool = False,
        data_token: Optional[str] = None,
        data_tokens: Optional[dict] = None,
        save_as: Optional[str] = None,
        load_from: Optional[str] = None,
    ) -> "LaunchConfig":
        """Configure the global launch settings.

        Args:
            residency: One of Constants.Residency values.
            endpoint: WebSocket server URL (required for edge/server).
                A ``wss`` scheme switches the data socket to TLS.
            bind: Which interfaces a server-residency broker listens on,
                one of Constants.Bind values. Separate from ``endpoint``
                because that host is the one clients dial.
            certfile: PEM certificate for a server-residency broker.
            keyfile: PEM private key, if not inside ``certfile``.
            ca_file: A certificate an edge should trust.
            allow_insecure: Permit a plaintext socket to a host that is
                not this machine.
            data_token: What an EDGE presents to the broker it dials.
                Falls back to ``GPYPE_DATA_TOKEN`` in the environment,
                which is how a process launched by an operator receives
                one -- deliberately not a command-line flag, because
                ``argv`` is visible in a process list.
            data_tokens: What a SERVER's broker accepts, keyed by
                capability (``put``, ``subscribe``). None keeps the
                historical behaviour of accepting every connection.
            save_as: Record every source's raw frames under this path
                stem. See :attr:`save_as`.
            load_from: Replay a recorded run instead of building any
                source's own core. See :attr:`load_from`.

        Returns:
            The configured LaunchConfig instance.

        Raises:
            ValueError: If residency is invalid or required params
                missing, or if both save_as and load_from are given.
            InsecureTransportError: If the endpoint is plaintext and
                names a host reachable from a network, and that was not
                asked for. The samples on this socket are the
                recording, so crossing a network in clear is a choice
                to be made rather than a default to fall into.
        """
        cls._validate_residency(residency, endpoint)
        cls._validate_bind(bind)
        if save_as and load_from:
            raise ValueError(
                "save_as and load_from cannot both be set: one records "
                "what the sources produce and the other replaces them "
                "with a recording, so together they would ask this "
                "process to record its own replay. Run them as two "
                "runs."
            )
        if residency != Constants.Residency.STANDALONE:
            check_endpoint_allowed(
                endpoint, allow_insecure, "exchange data with"
            )
            if (
                residency == Constants.Residency.SERVER
                and endpoint_is_secure(endpoint)
                and certfile is None
            ):
                raise ValueError(
                    f"endpoint {endpoint!r} asks for TLS, so the server "
                    "needs a certificate: pass certfile (and keyfile, "
                    "if the key is separate)."
                )

        if cls._instance is None:
            cls._instance = LaunchConfig()
        cls._instance.residency = residency
        cls._instance.endpoint = endpoint
        cls._instance.bind = bind
        cls._instance.certfile = certfile
        cls._instance.keyfile = keyfile
        cls._instance.ca_file = ca_file
        cls._instance.allow_insecure = allow_insecure
        # The environment, not a flag: a token on the command line is a
        # token in every process listing on the machine.
        cls._instance.data_token = data_token or os.environ.get(ENV_DATA_TOKEN)
        cls._instance.data_tokens = data_tokens
        cls._instance.save_as = save_as
        cls._instance.load_from = load_from

        # A stream's key is its class plus an ordinal in construction
        # order, so the counter has to start where the graph does.
        # Without this, a second pipeline in one process continues
        # counting and a load_from graph asks for 'Generator_1' from a
        # run that recorded 'Generator_0'. Imported here because the
        # module that owns the counter reads this one.
        _reset_raw_state()

        return cls._instance

    def server_tls(self) -> Optional[ServerTls]:
        """The certificate this process's broker should present, if any.

        Returns:
            A ServerTls when the endpoint asks for TLS, else None.
        """
        if not endpoint_is_secure(self.endpoint) or self.certfile is None:
            return None
        return ServerTls(certfile=self.certfile, keyfile=self.keyfile)

    def client_tls(self) -> Optional[ClientTls]:
        """How this process should verify the broker it connects to.

        Returns:
            A ClientTls when the endpoint asks for TLS, else None.
            ``allow_insecure`` here means "do not verify", which is a
            different thing from "do not encrypt" and is why it is one
            flag rather than two: a deployment that has said it trusts
            this network has said the same thing both times.
        """
        if not endpoint_is_secure(self.endpoint):
            return None
        return ClientTls(
            ca_file=self.ca_file,
            verify=not self.allow_insecure,
            check_hostname=not self.allow_insecure,
        )

    @classmethod
    def parse_args(
        cls,
        args: Optional[Sequence[str]] = None,
        parser: Optional[argparse.ArgumentParser] = None,
    ) -> "LaunchConfig":
        """Parse command-line arguments into launch configuration.

        No-op if LaunchConfig was already configured programmatically
        (i.e. configure() was called before this).

        Args:
            args: Command-line arguments. Defaults to sys.argv[1:].
            parser: Custom ArgumentParser to extend. If None, creates
                a new parser with standard g.Pype arguments.

        Returns:
            The configured LaunchConfig instance.

        Example:
            # In main()
            config = LaunchConfig.parse_args()
        """
        # Skip if already configured via configure()
        if cls._instance is not None:
            return cls._instance
        if parser is None:
            parser = argparse.ArgumentParser(
                description="g.Pype Application",
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )

        # Add standard g.Pype arguments
        cls._add_arguments(parser)

        # Parse arguments
        if args is None:
            args = sys.argv[1:]

        parsed, _ = parser.parse_known_args(args)

        # Extract args
        residency = getattr(
            parsed, "residency", Constants.Residency.STANDALONE
        )
        endpoint = getattr(parsed, "endpoint", None)

        return cls.configure(
            residency=residency,
            endpoint=endpoint,
            bind=getattr(parsed, "bind", Constants.Bind.ANY),
            certfile=getattr(parsed, "certfile", None),
            keyfile=getattr(parsed, "keyfile", None),
            ca_file=getattr(parsed, "ca_file", None),
            allow_insecure=getattr(parsed, "insecure", False),
            save_as=getattr(parsed, "save_as", None),
            load_from=getattr(parsed, "load_from", None),
        )

    @classmethod
    def _add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add standard g.Pype command-line arguments to parser."""
        parser.add_argument(
            "--residency",
            "-r",
            choices=[
                Constants.Residency.STANDALONE,
                Constants.Residency.EDGE,
                Constants.Residency.SERVER,
            ],
            default=Constants.Residency.STANDALONE,
            help="Execution mode: standalone (local), edge (data source), "
            "or server (processing). Default: standalone",
        )

        parser.add_argument(
            "--endpoint",
            "-e",
            type=str,
            default=None,
            help="WebSocket endpoint URL, e.g. wss://server:8765. "
            "Required for edge/server modes. A wss scheme "
            "encrypts the data socket; ws leaves it in clear and "
            "is refused for any host but this one unless "
            "--insecure is given.",
        )

        parser.add_argument(
            "--bind",
            choices=[
                Constants.Bind.ANY,
                Constants.Bind.LOOPBACK,
            ],
            default=Constants.Bind.ANY,
            help="Which interfaces a server-residency data socket "
            "listens on: any (every interface) or loopback (this "
            "machine only). Distinct from --endpoint, whose host is "
            "the address clients dial. Default: any",
        )

        parser.add_argument(
            "--certfile",
            type=Path,
            default=None,
            help="PEM certificate the broker presents. Required with a "
            "wss endpoint in server residency.",
        )

        parser.add_argument(
            "--keyfile",
            type=Path,
            default=None,
            help="PEM private key, if it is not inside --certfile.",
        )

        parser.add_argument(
            "--ca-file",
            type=Path,
            default=None,
            dest="ca_file",
            help="A certificate this process should trust in addition "
            "to the system store. How an edge verifies a "
            "self-signed server.",
        )

        parser.add_argument(
            "--insecure",
            action="store_true",
            help="Exchange data over a network in clear, or without "
            "verifying a certificate. The samples on this socket "
            "are the recording.",
        )

        parser.add_argument(
            "--save-as",
            type=str,
            default=None,
            dest="save_as",
            help="Record every source's raw frames, as they leave the "
            "source and before anything else sees them. Produces a "
            "directory of one file per stream plus a manifest.",
        )

        parser.add_argument(
            "--load-from",
            type=str,
            default=None,
            dest="load_from",
            help="Replay a run recorded with --save-as in place of "
            "every source, reproducing its block boundaries and "
            "timing. Takes the run directory.",
        )

    def bind_hosts(self):
        """The host argument a broker should bind for this config.

        Returns:
            ``None`` for :data:`Constants.Bind.ANY`, which asyncio binds
            as the wildcard on every interface and both address
            families; otherwise the loopback addresses this host has, as
            literals.

        Raises:
            ValueError: If ``bind`` is not a Constants.Bind value. Read
                access validates too, because the class documents
                assignment as a supported route -- and the failure this
                prevents is silent: any unrecognised value would
                otherwise fall through to the loopback branch, and a
                deployment that typed ``"ANY"`` would bind loopback and
                see every edge refused with nothing logged.

        The loopback case names ``127.0.0.1`` and ``::1`` rather than
        ``localhost`` on purpose. Binding the *name* would resolve it,
        and the resolution order is what cost every client ~2.0 s on a
        refused IPv6 attempt before this bind covered both families
        (measured: 2.021 s, against 0.002 s). Naming both literals keeps
        that property while refusing everything off this machine.
        """
        self._validate_bind(self.bind)
        if self.bind == Constants.Bind.ANY:
            return None
        hosts = ["127.0.0.1"]
        if socket.has_ipv6:
            hosts.append("::1")
        return hosts

    @classmethod
    def _validate_bind(cls, bind: str) -> None:
        """Validate the bind mode.

        Args:
            bind: The value to check.

        Raises:
            ValueError: If it is not a Constants.Bind value.
        """
        valid = (Constants.Bind.ANY, Constants.Bind.LOOPBACK)
        if bind not in valid:
            raise ValueError(f"Invalid bind: {bind}. Must be one of {valid}.")

    @classmethod
    def _validate_residency(
        cls,
        residency: str,
        endpoint: Optional[str],
    ) -> None:
        """Validate residency configuration."""
        valid = (
            Constants.Residency.STANDALONE,
            Constants.Residency.EDGE,
            Constants.Residency.SERVER,
        )
        if residency not in valid:
            raise ValueError(
                f"Invalid residency: {residency}. Must be one of {valid}."
            )

        if residency != Constants.Residency.STANDALONE:
            if endpoint is None:
                raise ValueError(
                    f"--endpoint required for {residency} residency"
                )

    @classmethod
    def reset(cls) -> None:
        """Reset to default configuration.

        Clears the raw-recording state too, so that ``reset()`` and
        ``configure()`` are symmetric. They were not: only ``configure``
        cleared the stream-key ordinals, the run directory and the
        replay anchor, so a host taking the documented third route --
        ``get()`` and then assign the fields -- kept the previous run's
        ordinals, and a pipeline would ask for ``Generator_1`` from a run
        that had recorded ``Generator_0``. That is exactly the drift the
        reset in ``configure`` exists to prevent.
        """
        cls._instance = None
        _reset_raw_state()

    @property
    def link_config(self) -> Optional[dict]:
        """Get link configuration dict for Source/Sink nodes.

        Returns:
            Dict with endpoint, or None for standalone.
        """
        if self.residency == Constants.Residency.STANDALONE:
            return None
        return {
            "endpoint": self.endpoint,
        }

    def is_standalone(self) -> bool:
        """Check if running in standalone mode."""
        return self.residency == Constants.Residency.STANDALONE

    def is_edge(self) -> bool:
        """Check if running in edge mode."""
        return self.residency == Constants.Residency.EDGE

    def is_server(self) -> bool:
        """Check if running in server mode."""
        return self.residency == Constants.Residency.SERVER
