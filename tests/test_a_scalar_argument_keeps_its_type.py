"""No template declares a scalar argument as `int` regardless of its type.

Measured 2026-09-12 on `addmm_kernel` at M=8 N=768 K=128 with a bf16 bias. The
emitted MSL carried

    device int* alpha_buf [[buffer(10)]];
    int alpha = alpha_buf[0];

while `alpha` is fp32. The bit pattern of `1.0f` is `0x3F800000` =
**1065353216**, and the measured deviation of our output against an fp64
reference was **1.065e+09** -- the same number, not a coincidence.

An independent third party settled who was wrong: ATen on MPS agreed with the
fp64 oracle to 5.5e-08 and disagreed with our kernel by that billion. The
screen had been refusing every candidate at this key, and it was RIGHT to.

The repair already existed, in ONE of the four sites that declare scalars, with
a comment describing this exact defect. Three others still emitted `int`, and
the one the simdgroup matmul takes was among them. A repair at ninety percent
leaves the next anomaly with the same cause and no trace of the first.

So the invariant is asserted over the FILE and not over the site that was
found, and by AST rather than text -- the comment explaining the defect
contains the very string a grep would look for.

Runnable: python3 -m pytest tests/test_a_scalar_argument_keeps_its_type.py -v
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

try:
    import triton_msl  # noqa: F401

    HAS = True
except Exception:                                     # pragma: no cover
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="triton_msl is not importable")

#: An emitted declaration that hard-codes `int` for a scalar buffer, or reads
#: one back as `int`. Both halves matter: declaring the buffer correctly and
#: reading it as int is the same defect one line later.
_HARDCODED = re.compile(r"device int\*\s*\{?\w*\}?\w*_buf|^\s*int \{arg\.name\}")


def _emitted_strings(path: Path):
    """Every string literal the module emits, with its line.

    By AST: the comment that explains this defect contains `device int*`, and
    a text search would report the file broken for as long as the explanation
    survives -- or be loosened until it saw nothing.
    """
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append((node.lineno, node.value))
        elif isinstance(node, ast.JoinedStr):
            out.append((node.lineno, ast.unparse(node)))
    return out


def _templates_file():
    import triton_msl.codegen._lowerer_templates as T
    return Path(T.__file__)


@requires
def test_no_template_declares_a_scalar_buffer_as_int():
    bad = [(n, s.strip()[:80]) for n, s in _emitted_strings(_templates_file())
           if "device int*" in s and "_buf" in s]
    assert not bad, (
        "these sites declare a scalar argument's buffer as int whatever its "
        "real type:\n  " + "\n  ".join(f"line {n}: {s}" for n, s in bad)
        + "\n\nA scalar is declared with ITS OWN type. `alpha` and `beta` are "
          "fp32; read as int their bit pattern becomes 1065353216.")


@requires
def test_no_template_reads_a_scalar_back_as_int():
    bad = [(n, s.strip()[:80]) for n, s in _emitted_strings(_templates_file())
           if re.search(r"\bint \{arg\.name\}", s) and "_buf[0]" in s]
    assert not bad, (
        "these sites read a scalar back as int:\n  "
        + "\n  ".join(f"line {n}: {s}" for n, s in bad)
        + "\n\nDeclaring the buffer correctly and reading it as int is the "
          "same defect one line later.")


@requires
def test_the_guard_can_see_one():
    """Both directions, with a decoy in a comment.

    A guard that reports zero over a file must be shown able to report one,
    and shown not to be fooled by the prose that explains why it is zero.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write('# device int* alpha_buf in a comment must NOT count\n'
                'def emit(arg, i):\n'
                '    return f"    device int* {arg.name}_buf [[buffer({i})]]"\n')
        tmp = Path(f.name)
    try:
        found = [s for _, s in _emitted_strings(tmp)
                 if "device int*" in s and "_buf" in s]
        assert len(found) == 1, f"expected the f-string only, got {found}"
    finally:
        tmp.unlink()
