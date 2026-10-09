"""Season 8, Episode 2: a serialized pipeline, loaded by another process.

The pipeline uses a node from a package of your own, which this script
writes to a temporary directory. Every load runs in a fresh process,
because the allow-list is read once per process.
"""

import json
import os
import subprocess
import sys
import tempfile

import gpype as gp
from gpype.common.document import node_ids

#: Your node, in your package. It imports a helper from a second
#: package of yours, which matters once the code travels in a bundle.
MY_NODES = '''
import gpype as gp
from my_helpers import scale


class Gain(gp.IONode):
    """Multiplies every sample by a factor."""

    class Configuration(gp.IONode.Configuration):
        class Keys(gp.IONode.Configuration.Keys):
            FACTOR = "factor"

    def __init__(self, factor=2.0, **kwargs):
        super().__init__(factor=factor, **kwargs)

    def step(self, data):
        return {"out": scale(data["in"], self.config["factor"])}
'''

MY_HELPERS = '''
def scale(block, factor):
    return block * factor
'''

#: What the loading process sees of its environment: nothing permitted.
CLEAN = {k: v for k, v in os.environ.items()
         if not k.startswith("GPYPE_ALLOWED_MODULES")}


def write_package(home: str, name: str, source: str) -> None:
    """Write a one-file package into *home*."""
    os.makedirs(os.path.join(home, name))
    with open(os.path.join(home, name, "__init__.py"), "w") as handle:
        handle.write(source)


def save(home: str, name: str, document: dict) -> str:
    """Write *document* as JSON into *home*, and return its path."""
    path = os.path.join(home, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    return path


def load(path: str, allow: bool) -> None:
    """The other process: read a serialized pipeline and rebuild it."""
    if allow:
        # The host program vouches for its own package, in its own code.
        gp.allow_modules("my_nodes.*")
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    try:
        p = gp.Pipeline.deserialize(document)
    except Exception as error:
        print(f"{type(error).__name__}: {error}")
        return
    print("loaded:", repr(p))
    p.close()


def elsewhere(path: str, env: dict, allow: bool = False) -> None:
    """Load *path* in a fresh process, with *env* added to its own."""
    command = [sys.executable, __file__, "load", path]
    if allow:
        command.append("allow")
    sys.stdout.flush()
    subprocess.run(command, env={**CLEAN, **env})


if __name__ == "__main__" and sys.argv[1:2] == ["load"]:
    load(sys.argv[2], allow=sys.argv[3:] == ["allow"])

elif __name__ == "__main__":

    HOME = tempfile.mkdtemp(prefix="s8e2_")
    write_package(HOME, "my_nodes", MY_NODES)
    write_package(HOME, "my_helpers", MY_HELPERS)
    sys.path.insert(0, HOME)
    from my_nodes import Gain

    p = gp.Pipeline()
    source = gp.Generator(sampling_rate=250, channel_count=2,
                          signal_amplitude=10.0, signal_frequency=10)
    gain = Gain(factor=3.0, name="gain")
    p.connect(source, gain)
    p.connect(gain, gp.Collector())

    # ---- 1. A serialized pipeline -------------------------------------
    document = p.serialize()
    print("modules  :", [node["module"] for node in document["nodes"]])
    print("ids      :", node_ids(document))
    # One written by hand or by a tool may leave an id out. Fill it
    # once, before it goes anywhere, since two processes left to
    # invent it would invent two different ones.
    edited = json.loads(json.dumps(document))
    del edited["nodes"][2]["config"]["id"]
    print("edited   :", node_ids(edited))
    print("filled   :", node_ids(gp.canonicalize(edited)))

    plain = save(HOME, "plain.json", document)
    listed = os.path.join(HOME, "allowed.txt")
    with open(listed, "w") as handle:
        handle.write("# modules this deployment trusts\nmy_nodes.*\n")

    # Your packages, installed where the pipeline is loaded.
    path = filter(None, [HOME, os.environ.get("PYTHONPATH")])
    installed = {"PYTHONPATH": os.pathsep.join(path)}
    permit = {"GPYPE_ALLOWED_MODULES": "my_nodes.*"}

    # ---- 2. Refused, then three ways to permit it ---------------------
    print("\n--- another process, nothing permitted")
    elsewhere(plain, installed)
    print("\n--- GPYPE_ALLOWED_MODULES")
    elsewhere(plain, {**installed, **permit})
    print("\n--- GPYPE_ALLOWED_MODULES_FILE")
    elsewhere(plain, {**installed, "GPYPE_ALLOWED_MODULES_FILE": listed})
    print("\n--- allow_modules(), in the host program")
    elsewhere(plain, installed, allow=True)

    # ---- 3. A host that does not have your packages -------------------
    print("\n--- permitted, but not installed there")
    elsewhere(plain, permit)
    print("\n--- my_nodes bundled, my_helpers not")
    one = p.serialize(packages=["my_nodes"])
    elsewhere(save(HOME, "one.json", one), permit)
    print("\n--- both bundled")
    both = p.serialize(packages=["my_nodes", "my_helpers"])
    elsewhere(save(HOME, "both.json", both), permit)
    p.close()
