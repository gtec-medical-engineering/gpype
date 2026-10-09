"""What this process is, as g.Pype's licence gate sees it.

A tool that builds or supports a g.Pype application reads this rather than
g.Pype's private modules. gpype-compiler checks a built application with
it, and its startup hook opens the licence window only when
``deployment_report()["licence"]["permission"]`` is ``abort``, which is
when the gate would refuse a run. The hook also binds the application's
own licence product, with ``set_licence_product()``.
"""

from __future__ import annotations

from typing import Any, Dict


def deployment_report() -> Dict[str, Any]:
    """Describe this process as the licence gate would judge it now.

    Reading it decides nothing and latches nothing; ``Pipeline.start()``
    still makes the decision for a run. The licence is read from the local
    store, with no network. The residency is read as the gate will read
    it, and the process-wide ``LaunchConfig`` is left as it was found: a
    report taken before an application configures itself must not fix
    that configuration from the command line.

    Returns:
        A JSON-serialisable dict:

        * ``gpype_version``, ``api_version``
        * ``frozen``: whether the process is a frozen application, read
          from the operating system rather than from ``sys``.
        * ``process``: one line on how that was decided, for a support
          question.
        * ``residency``: ``standalone``, ``edge`` or ``server``.
        * ``licence``: ``product``, the product the gate asks about;
          ``state``, one of ``licensed``, ``unlicensed`` or ``unknown``;
          ``detail``, the underlying verdict; ``permission``, one of
          ``full``, ``limited`` or ``abort``, what the licence gate (G1)
          would grant; ``reason``, why, empty when ``full``.

        It does not say what a run would be granted. The device gate (G2)
        is decided at ``start()`` from the pipeline's own sources, and a
        run takes the stricter of the two: a fully licensed pipeline whose
        sources attest nothing, a ``Generator`` say, still runs marked.
    """
    from .. import API_VERSION, __version__
    from ._private import deployment, licence
    from .launch_config import LaunchConfig

    state, detail = licence.query_licence()
    # The name the gate itself calls, so the two cannot drift apart.
    frozen = licence.is_frozen_deployment()
    configured = LaunchConfig._instance
    try:
        residency = licence._residency()
    finally:
        # _residency() builds the singleton from argv when there is none;
        # a later LaunchConfig.parse_args() would then do nothing.
        LaunchConfig._instance = configured
    permission, reason = licence.evaluate(
        frozen=frozen, residency=residency, licence=(state, detail)
    )
    return {
        "gpype_version": __version__,
        "api_version": API_VERSION,
        "frozen": frozen,
        "process": deployment.describe(),
        "residency": str(residency),
        "licence": {
            "product": licence.product(),
            "state": state,
            "detail": detail,
            "permission": permission.name.lower(),
            "reason": reason,
        },
    }


def set_licence_product(product: str) -> None:
    """Make the licence gate ask about an application's own product.

    By default the gate asks about "g.Pype Runtime". An application
    licensed under a product of its own binds it here, before its first
    pipeline starts; gpype-compiler's startup hook does so for a build
    that names one. One product per process, and never one of g.tec's
    existing licences or a name in one of its product families, such as
    "Unicorn ..." (D-ENT-101, D-ENT-103).

    Args:
        product: The licence product, as the licence server names it.

    Raises:
        ValueError: If the name is empty, names an existing g.tec
            licence, or begins with a g.tec product family.
        RuntimeError: If this process already bound another product.
    """
    from ._private import licence

    licence.bind_product(product)
