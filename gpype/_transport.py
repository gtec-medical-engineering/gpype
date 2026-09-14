"""Transport security for the sockets g.Pype and its runtime open.

A private contract, like :mod:`gpype._wire` beside it, and here for the
same reason: there are two sockets between an edge and a server -- the
runtime's control socket and g.Pype's own data socket -- and they have to
agree on when encryption is required. Written once, in the package that
is the other's dependency, rather than twice; ``gpype_runtime.common.tls``
re-exports these names.

Encrypting only the control plane would have been the worse outcome of
the two: the control socket carries credentials, but the data socket
carries the recording.

Two rules, and they are the whole policy:

* **A plaintext socket to anywhere but the local machine has to be asked
  for.** Not a warning -- a refusal. A LAN or cloud deployment that
  forgets to configure a certificate should fail to start rather than
  quietly ship credentials and signals in clear, so the default is the
  safe one and opting out is explicit.
* **Verification is not optional unless it is named.** A self-signed
  certificate is fine; the peer trusts it by being pointed at it
  (``ca_file``). Turning verification off entirely is a separate,
  explicitly-named choice, because TLS without verification stops any
  passive listener and no active one.

Loopback is the one place plaintext is the default: the traffic does not
reach a network, and requiring a certificate to run a pipeline on your
own machine would only teach everyone the opt-out flag.
"""

from __future__ import annotations

import ipaddress
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

#: Hostnames that mean this machine without going near a network.
_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", ""})

#: The scheme that means "encrypted" for a WebSocket URL.
SECURE_SCHEME = "wss"

#: And the one that means it is not.
PLAIN_SCHEME = "ws"


def is_loopback(host: Optional[str]) -> bool:
    """Whether traffic to ``host`` stays on this machine.

    Args:
        host: A hostname or address, as configured.

    Returns:
        True for loopback names and addresses. False for anything else,
        including the wildcard binds ``0.0.0.0`` and ``::`` -- a wildcard
        accepts connections from the network, which is the case this
        exists to catch.
    """
    if host is None:
        return True
    name = host.strip().lower()
    if name.startswith("[") and name.endswith("]"):
        # As it appears in a URL's authority.
        name = name[1:-1]
    if name in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def endpoint_is_secure(endpoint: Optional[str]) -> bool:
    """Whether an endpoint URL asks for TLS.

    The scheme is the switch, because both ends read the same endpoint
    out of the same configuration: an edge and a server cannot disagree
    about whether the socket between them is encrypted if the answer is
    written in the address they both use.

    Args:
        endpoint: A ``ws://`` or ``wss://`` URL, or None.

    Returns:
        True only for ``wss``.
    """
    if not endpoint:
        return False
    return urlparse(endpoint).scheme.lower() == SECURE_SCHEME


def host_of(endpoint: Optional[str]) -> Optional[str]:
    """The host an endpoint URL names, or None.

    Args:
        endpoint: A WebSocket URL, or None.

    Returns:
        The hostname, lowercased by urlparse, or None if the URL names
        none.
    """
    if not endpoint:
        return None
    return urlparse(endpoint).hostname


class InsecureTransportError(RuntimeError):
    """Raised when a plaintext socket was neither safe nor asked for."""


@dataclass(frozen=True)
class ServerTls:
    """The certificate a listener presents.

    Attributes:
        certfile: PEM certificate, or a chain ending in one.
        keyfile: PEM private key. None if the key is in ``certfile``.
        password: Passphrase for an encrypted key, if it has one.
    """

    certfile: Path
    keyfile: Optional[Path] = None
    password: Optional[str] = None

    def context(self) -> ssl.SSLContext:
        """Build the context to serve with.

        Returns:
            A server context holding this certificate.

        Raises:
            FileNotFoundError: If a configured file is not there. Raised
                here rather than surfacing later as a handshake failure
                on the peer's side, which is the wrong place to read a
                typo in a path.
        """
        for path in (self.certfile, self.keyfile):
            if path is not None and not Path(path).is_file():
                raise FileNotFoundError(f"No certificate file at {path}")
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(
            certfile=str(self.certfile),
            keyfile=str(self.keyfile) if self.keyfile else None,
            password=self.password,
        )
        return context


@dataclass(frozen=True)
class ClientTls:
    """What a connecting peer will accept from a listener.

    Attributes:
        ca_file: A certificate or CA bundle to trust in addition to the
            system store. This is how a self-signed deployment is
            verified rather than waved through.
        verify: Whether to verify the peer's certificate at all. Setting
            this False is a deliberate, named choice: it leaves the
            connection encrypted against a passive listener and open to
            an active one.
        check_hostname: Whether the certificate has to name the host
            being connected to. Only meaningful when ``verify``.
    """

    ca_file: Optional[Path] = None
    verify: bool = True
    check_hostname: bool = True

    def context(self) -> ssl.SSLContext:
        """Build the context to connect with.

        Returns:
            A client context, verifying unless told not to.

        Raises:
            FileNotFoundError: If ``ca_file`` is not there. Otherwise a
                typo in the path would silently fall back to the system
                store and refuse a certificate that was configured
                correctly.
        """
        if self.ca_file is not None and not Path(self.ca_file).is_file():
            raise FileNotFoundError(
                f"No certificate authority file at {self.ca_file}"
            )
        context = ssl.create_default_context(
            ssl.Purpose.SERVER_AUTH,
            cafile=str(self.ca_file) if self.ca_file else None,
        )
        if not self.verify:
            # Order matters: check_hostname must go first, or setting
            # verify_mode to CERT_NONE raises.
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        else:
            context.check_hostname = self.check_hostname
        return context


def scheme_for(tls: Optional[object]) -> str:
    """The URL scheme that matches a TLS configuration.

    Args:
        tls: A ServerTls, a ClientTls, or None.

    Returns:
        ``"wss"`` if TLS is configured, ``"ws"`` otherwise. Derived
        rather than written down, because several places used to hardcode
        ``ws://`` and would have gone on saying so after TLS was switched
        on.
    """
    return SECURE_SCHEME if tls is not None else PLAIN_SCHEME


#: What the control socket would expose, which is the default because
#: it is the socket a credential crosses.
EXPOSES_CREDENTIALS = "the shared secret, the session id and the session key"

#: And what the data socket would expose. Different words for a
#: different risk: one is an account, the other is a recording.
EXPOSES_SIGNALS = "every sample of the recording, and the pipeline itself"


def check_plaintext_allowed(
    host: Optional[str],
    allow_insecure: bool,
    what: str,
    exposes: str = EXPOSES_CREDENTIALS,
) -> None:
    """Refuse a plaintext socket that was neither safe nor asked for.

    Args:
        host: The host being bound or connected to.
        allow_insecure: Whether plaintext was explicitly chosen.
        what: How to name the socket in the message.
        exposes: What would be readable, named in the message. An
            operator deciding whether to pass the flag is deciding about
            a specific risk, and "the session key" and "the recording"
            are not the same decision.

    Raises:
        InsecureTransportError: If ``host`` is reachable from a network
            and plaintext was not asked for.
    """
    if allow_insecure or is_loopback(host):
        return
    raise InsecureTransportError(
        f"Refusing to {what} {host!r} without TLS: {exposes} would "
        "cross the network in clear. Configure a certificate, or pass "
        "allow_insecure=True to say that this network is trusted."
    )


def check_endpoint_allowed(
    endpoint: Optional[str],
    allow_insecure: bool,
    what: str,
    exposes: str = EXPOSES_SIGNALS,
) -> None:
    """The same refusal, for a socket named by a URL.

    Args:
        endpoint: A ``ws://`` or ``wss://`` URL, or None.
        allow_insecure: Whether plaintext was explicitly chosen.
        what: How to name the socket in the message.
        exposes: What would be readable. Defaults to the data socket's
            answer, because that is the socket named by an endpoint.

    Raises:
        InsecureTransportError: If the endpoint is plaintext and names a
            host reachable from a network.
    """
    if endpoint_is_secure(endpoint):
        return
    check_plaintext_allowed(host_of(endpoint), allow_insecure, what, exposes)
