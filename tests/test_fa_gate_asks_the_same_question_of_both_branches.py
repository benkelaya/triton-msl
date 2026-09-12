"""The FlashAttention scale/bias gate asks one question, of every operator.

The gate refuses an elementwise scale or bias applied to a `tt.dot` RESULT,
because the generic attention lowering drops or mis-applies it. Its own text
names the form: `qk = tl.dot(q, kT); qk = qk * (1/sqrt(d))` -- a SCALAR.

Its add branch asked whether the other operand is a scalar splat. Its multiply
branch returned True unconditionally, on the reasoning that a multiplication of
a dot result is "always a scale". It is not. Online-softmax FlashAttention
rescales its accumulator by a per-row VECTOR on every iteration:

    acc = acc * exp(m_i - m_ij)[:, None]

which is a multiply consuming a dot whose other operand is a tensor. So the
gate refused the canonical FlashAttention, while its own comment claimed "the
validated FA is unaffected; only a true scale/bias on the scores is".

Measured 2026-09-12 on whisper-large-v3-turbo: two dots, one `arith.mulf`
consuming a dot, its other operand produced by `tt.broadcast`, and
`_is_scalar_splat` FALSE on it. Three guesses at which kernel carried a scale
on the scores had all been wrong, because none did.

Runnable: python3 -m pytest tests/test_fa_gate_asks_the_same_question_of_both_branches.py -v
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


def _gate_source() -> str:
    import triton_msl.codegen.generic_lowerer as G

    src = Path(G.__file__).read_text()
    start = src.index("def _dot_result_scaled(_dot):")
    end = src.index("if any(_dot_result_scaled(", start)
    return src[start:end]


@requires
def test_the_gate_exists_at_all():
    """Without this the two below pass over an absent function."""
    assert "def _dot_result_scaled" in _gate_source()


@requires
def test_no_operator_returns_true_without_asking_about_the_operand():
    """Every `return True` in the gate must be governed by the splat test.

    The defect was an asymmetry between two branches doing the same job, and
    an asymmetry is invisible to a test that exercises only one of them. So
    the assertion is on the SHAPE of the decision: no unconditional yes.
    """
    tree = ast.parse("def f():\n" + "\n".join(
        "    " + l for l in _gate_source().splitlines()[1:] if l.strip()))
    bare = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        cond = ast.unparse(node.test)
        if "_s.op in" not in cond:
            continue
        body = ast.unparse(node)
        if "_is_scalar_splat" not in body:
            bare.append(cond[:70])
    assert not bare, (
        "these operator branches return a verdict without asking whether the "
        "other operand is a scalar splat:\n  " + "\n  ".join(bare)
        + "\n\nA multiply consuming a dot is not always a scale: the online "
          "softmax rescales its accumulator by a per-row vector.")


@requires
def test_the_splat_test_is_what_both_branches_use():
    """And it is the SAME test, not two that could drift.

    Two copies of one question is one that can be fixed alone -- which is
    exactly how the asymmetry arose.
    """
    body = _gate_source()
    assert body.count("_is_scalar_splat") >= 1
    assert len(re.findall(r"def _is_scalar_splat", body)) == 0, (
        "the splat test must be defined once, outside this function, not "
        "redefined per branch")
