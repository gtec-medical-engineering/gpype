"""The widget half of each scope, imported only where it is built.

A scope is an ``IChain`` whose internal nodes are ``[Link, _XCore]``, and
in server residency the core is never built -- there is no display on a
server. The core classes nonetheless used to live beside their chains, so
*importing* a scope's module imported Qt: a document naming a scope pulled
PySide6 into a headless server process that never draws anything, which is
the difference between a container image that needs a GUI toolkit and one
that does not.

They live here instead, and a chain imports its core inside the residency
branch that constructs it. Private: nothing outside ``widgets`` should
reach for these, and a chain is the supported way to use one.
"""
