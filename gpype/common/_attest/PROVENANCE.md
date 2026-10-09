# Vendored attestation verifier

## Why this is vendored rather than imported

g.Pype depends on `gtec_attest`, and could simply `import` it. That is the
form WP-E4 rejected: an importable Python module is **shadowable**. A
five-line stub earlier on `sys.path` —

```python
# gtec_attest.py, anywhere ahead of site-packages
class _V:
    verified = True
    class reason:
        name = "OK"
def verify(*a, **k):
    return _V()
def mint_nonce():
    return b"\x00" * 32
```

— defeats device attestation without touching a single compiled byte. It
is the same attack that forces the Unicorn wrapper to read its serial from
`Unicorn.dll` rather than through `UnicornPy` (WP-B6-ii).

So the verifier is compiled **into** g.Pype's own compiled unit. The
upstream header says this is what it is for:

> These are intended to be compiled directly into the enforcing product's
> own compiled unit, so that the check cannot be replaced by shadowing an
> importable Python module on `sys.path`.

`gpype.common._attest_native` is a submodule of a package, so it resolves
through `gpype.__path__` rather than `sys.path`: replacing it means
writing into the installed package, not dropping a file in the working
directory. That is a materially higher bar, and it is the property being
bought here.

**Accepted cost, per WP-E4:** key rotation becomes a rebuild of every
consumer rather than a package bump. Re-vendor and rebuild when
`gtec-attest` rotates.

## Upstream

- **Source:** `gtec-attest`, `src/gtec_attest/core/`
- **Retrieved:** 2026-08-21; `gtec_attest_keys.h` again on 2026-10-05,
  from `gtec-attest-core-1.0.0.zip`, the core zip of the gtec_attest
  1.0.0 release (`6f307ab`), which retires `0x0002` and `0x0003`. The
  other six files are byte-identical to that zip and to 0.4.0's, so all
  seven hashes below are the release's
- **Licence:** g.tec Non-Commercial (GNCL) for the g.tec files; TweetNaCl
  is public domain
- **Contains no secrets.** Public keys only: `0x0004`, the amplifier key,
  whose private half is in the amplifier drivers' release builds;
  `0x0005`, reserved for licence attestations signed by g.tec's licence
  server, which no build carries; and three retired keys -- `0x0001`, the
  first amplifier key; `0x0003`, the amplifier key before `0x0004`; and
  `0x0002`, gtec_licensing's licence key up to 2.2.0 (D-ENT-102). The
  header marks all three retired, and so does `attestation.KEY_DOMAINS`,
  so a table that listed one as active still could not vouch with it
  (D-ENT-73). The header's comments are
  upstream's registry notes, rendered, and are not edited here. Each of
  the five arrays is 32 bytes, one public key, not 64.

**The table is domain-blind.** It trusts every key it lists, and they
all sign the same payload shape, so `attestation.verifier(domain)` refuses
a key from another domain (D-ENT-70) and a retired one in every domain
(D-ENT-73). A key added upstream is trusted nowhere until
`attestation.KEY_DOMAINS` names its domain.

SHA-256 as vendored:

```
9b6b8c2d44fb7b62809659c8ac80f3a7ab177c9f3fd24053bc00a5d8105c3135  gtec_attest_verify.c
744fda421f81179decaa658340f2b8b101571bfb3caccf13f35114c0d8261e6a  gtec_attest_verify.h
315e6eb084263568e1bf28cb6b661c031d6e2f5365c288b3e908cf4cb0404e52  gtec_attest_keys.h
02e65bc3013ff2168983365e55906bc783c4c7e0a60d8100f17bb303a17175c4  tweetnacl.c
43f29ad721d9927b747b0100ab4160c119e7bb180c7c98a66e4bf79d31244287  tweetnacl.h
bb75a323df5fe149a73cee3d5d0be565ec7a02aa2e0a196b0c8dbdbc6953e7d5  randombytes_stub.c
76f5553a0150ff8710ffbc1052d52ef4b40147adeef030bbef46ce66cc0c8853  gtec_attest.pxd
```

## One Ed25519, three places

The `tweetnacl.[ch]` hashes above are **byte-identical** to the copies in
`gtec-ble` (`src/python/gtec_ble/lib/_attest/`) and in `gtec-attest`
itself. That is deliberate and worth preserving: the amplifier that
*signs*, the package that *verifies*, and g.Pype's linked verifier all run
the same implementation, which removes a class of "the two sides
disagree" bug that is miserable to diagnose across three repositories.

If `gtec-attest` re-vendors TweetNaCl, re-copy here rather than patching,
and re-copy in `gtec-ble` too.

## The upstream `.pxd`, and why it is here but not used

`gtec_attest.pxd` is vendored too, and `test_attest_linkage.py` checks
that the `cdef extern` block in `_attest_native.pyx` still matches it.

Upstream *intends* consumers to `cimport` it:

```
from gtec_attest cimport gtec_attest_verify, gtec_attest_key_t
```

which would remove the duplicate declaration entirely. That is not done
here on purpose. `setup.py` does not call `cythonize()` explicitly --
setuptools compiles the `.pyx` implicitly -- so getting a `cimport` to
resolve means relying on `include_dirs` reaching Cython as well as the C
compiler, across five Python versions and three platforms. The gain is
removing a duplicated declaration; the risk is a build that fails
somewhere in the matrix for a reason nobody will enjoy diagnosing.

So the declaration is written out, and drift is caught by a test instead
of prevented by the build. If `setup.py` ever calls `cythonize()` with an
explicit `include_path`, switch to the `cimport` and delete both the
extern block and that test.

**Note on the supported channel.** Upstream ships the linkable core as a
separate `gtec-attest-core-<ver>.zip` GitHub release artifact -- with an
`INTEGRATION.md` -- and deliberately keeps `.c`, `.h` and `.pxd` out of
the published wheel. The first copies were taken from the repository
working tree, before `gtec_attest` had releases (WP-E6); since 0.2.0 all
seven match the core zip. **Re-vendor from the core zip**, and check the
hashes here against it.

## What is used

Only `gtec_attest_verify()` — a pure function: no I/O, no clock, no
network, no allocation, no global state. `randombytes` is stubbed and
aborts if called, because verification needs no entropy. Nonce minting
stays in Python (`os.urandom`), where it is a one-liner and not the part
an attacker gains anything by replacing: a forged *nonce* only lets you
replay your own challenge, whereas a forged *verdict* clears any device.

## Verify at any time

```
sha256sum src/gpype/common/_attest/*.c src/gpype/common/_attest/*.h
```

`test_attest_linkage.py` does this automatically and fails if a vendored
file drifts from the hash recorded here.
