"""The wrap loop must sit INSIDE the k-loops, and the phases in one order.

Written before the lowerer is touched, and driven by the IR a model actually
produces rather than by a kernel written to resemble it. Three attempts at the
resemblance route -- a matmul in a k-loop, then a wider one, then one with a
div/rem row index -- all compiled cleanly, each through a template the real
kernel never reaches. That is measuring a proxy, and the first test below
exists so it cannot happen silently again.

`tests/golden/staged_dot_in_nested_loops.ttgir` is the TTGIR captured at the
exact point the refusal is raised, during a convolution on an upscaler, with
its source path rewritten. It reproduces the refusal standalone. Its shape:

    scf.for                      # three nested loops
      scf.for
        scf.for
          tt.load(ptr, mask, other)   -> ttg.local_alloc
          tt.load(ptr, mask, other)   -> ttg.local_alloc
          tt.dot

with a 2048-element tile. Both staged operands are MASKED loads, which is why
the mask had to come first: the fill that drops a mask reads out of bounds, and
the fill that substitutes zero for a declared `other` gives the dot a tile the
kernel never asked for. Neither shows in a value until the nesting lets a value
exist at all.

What this file pins is the NESTING, and the failure it is aimed at is not the
loud one. Emitting the cooperative loop at the wrong brace depth is a compile
error and announces itself. Three things do not announce themselves:

  * a `threadgroup_barrier` inside the per-element wrap loop is UNDEFINED, and
    undefined on this hardware means "usually right";
  * the phases emitted in the wrong order -- the dot reading shared memory the
    fill has not written yet -- gives a wrong number, not an error;
  * the barrier omitted between fill and dot gives a wrong number on some runs
    and the right one on others.

So the structural assertions below are written against emitted MSL, and each
one is paired with a synthetic counter-example it must REJECT. The pairing is
the point: an assertion that only ever confirms is satisfied by a detector that
says yes to anything, and this file exists because that has already happened
twice in this tree.

Runnable: python3 -m pytest tests/test_staged_dot_nesting_phases.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import triton
    import triton.language as tl
    from triton.compiler.compiler import compile as triton_compile, ASTSource
    from triton.backends.compiler import GPUTarget
    from triton_msl.errors import MetalNonRecoverableError

    HAS = True
except Exception:                                     # pragma: no cover
    HAS = False

requires_triton = pytest.mark.skipif(not HAS, reason="triton + triton_msl needed")

#: The refusal this file is aimed at, by its own words.
_NESTING_REFUSAL = "per-element wrap loop is emitted OUTSIDE that loop"

#: The cooperative fill loop and the per-element wrap loop, as emitted.
_WRAP = re.compile(r"for\s*\(\s*(?:u?int\w*\s+)?_loop_e\s*=")
_FILL = re.compile(r"for\s*\(\s*(?:u?int\w*\s+)?_sa\s*=")
_BARRIER = "threadgroup_barrier"


#: The captured IR. A path, not an ASTSource: triton's `compile` builds the
#: IRSource itself and asserts on a plain string for anything that is not AST.
FIXTURE = str(Path(__file__).parent / "golden" / "staged_dot_in_nested_loops.ttgir")


# The compilation caches belong to the session, enforced in
# `tests/conftest.py`: a session that cannot own them does not start. A
# per-module copy of that fixture lived here first and is gone, because two
# guards for one property is one that can be fixed alone.


def _compile():
    return triton_compile(FIXTURE, target=GPUTarget("metal", "apple-m4", 32))


# ── the detectors, proven on synthetic MSL before they are trusted ─────────


def barrier_inside_wrap(msl: str) -> bool:
    """Is any `threadgroup_barrier` inside the body of a `_loop_e` loop?

    Brace-counted from the wrap loop's opening brace, because a barrier two
    statements after the loop and a barrier inside it read the same to a regex
    that only asks whether both strings are present.
    """
    for m in _WRAP.finditer(msl):
        depth = 0
        started = False
        for i in range(m.start(), len(msl)):
            ch = msl[i]
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    break
            elif started and msl.startswith(_BARRIER, i):
                return True
    return False


def phase_order(msl: str) -> list[str]:
    """The phases in emission order: `fill`, `barrier`, `dot`.

    `dot` is taken as the first read of the staged buffer AFTER a fill, not as
    the word `dot`, which does not appear in the emitted text.
    """
    events = []
    for m in _FILL.finditer(msl):
        events.append((m.start(), "fill"))
    for m in re.finditer(re.escape(_BARRIER), msl):
        events.append((m.start(), "barrier"))
    for m in re.finditer(r"for\s*\(\s*(?:u?int\w*\s+)?_de\s*=", msl):
        events.append((m.start(), "dot"))
    return [name for _, name in sorted(events)]


_GOOD = """
kernel void k() {
  for (uint k_0 = 0; k_0 < K; ++k_0) {
    for (uint _loop_e = lid; _loop_e < 2048u; _loop_e += 1024u) { acc[_loop_e] = 0; }
    for (uint _sa = lid; _sa < 1024u; _sa += 1024u) { smem[_sa] = a[_sa]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint _de = lid; _de < 2048u; _de += 1024u) { acc[_de] += smem[_de]; }
  }
}
"""

_BARRIER_IN_WRAP = """
kernel void k() {
  for (uint _loop_e = lid; _loop_e < 2048u; _loop_e += 1024u) {
    for (uint k_0 = 0; k_0 < K; ++k_0) {
      for (uint _sa = lid; _sa < 1024u; _sa += 1024u) { smem[_sa] = a[_sa]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}
"""

_DOT_BEFORE_FILL = """
kernel void k() {
  for (uint k_0 = 0; k_0 < K; ++k_0) {
    for (uint _de = lid; _de < 2048u; _de += 1024u) { acc[_de] += smem[_de]; }
    for (uint _sa = lid; _sa < 1024u; _sa += 1024u) { smem[_sa] = a[_sa]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
"""

_NO_BARRIER = """
kernel void k() {
  for (uint k_0 = 0; k_0 < K; ++k_0) {
    for (uint _sa = lid; _sa < 1024u; _sa += 1024u) { smem[_sa] = a[_sa]; }
    for (uint _de = lid; _de < 2048u; _de += 1024u) { acc[_de] += smem[_de]; }
  }
}
"""


def test_the_barrier_detector_separates_inside_from_after():
    """Both directions, or the assertion on real MSL means nothing."""
    assert barrier_inside_wrap(_BARRIER_IN_WRAP) is True, (
        "a barrier nested inside the wrap loop's braces must be SEEN — it is "
        "undefined behaviour, and undefined on this hardware means it usually "
        "gives the right answer, which is the worst way for it to fail")
    assert barrier_inside_wrap(_GOOD) is False, (
        "a barrier that follows the wrap loop must not be flagged; a detector "
        "that cannot tell them apart would refuse the correct emission")


def test_the_phase_order_detector_sees_a_swap():
    assert phase_order(_GOOD) == ["fill", "barrier", "dot"]
    assert phase_order(_DOT_BEFORE_FILL)[0] == "dot", (
        "the dot reading shared memory before the fill writes it is a wrong "
        "number, not an error; the detector must order the phases by position")
    assert "barrier" not in phase_order(_NO_BARRIER), (
        "a missing barrier between fill and dot is a race: right on some runs "
        "and wrong on others. It must be visible as an absence.")


# ── the refusal today, and what must replace it ───────────────────────────


@requires_triton
def test_the_nesting_refusal_is_actually_reached_by_this_shape():
    """Without this, every assertion below passes over a kernel that never
    took the generic path at all — which is what the first version of this
    file did, compiling cleanly through the 128-thread matmul template.
    """
    with pytest.raises(MetalNonRecoverableError) as exc:
        _compile()
    assert _NESTING_REFUSAL in str(exc.value), (
        f"this shape must reach the NESTING refusal, not some other one. "
        f"Got: {str(exc.value)[:200]}")


@requires_triton
@pytest.mark.xfail(strict=True, reason="the nesting is not served yet: the "
                   "wrap loop is still emitted outside the k-loops. Remove "
                   "this marker with the fix, not before.")
def test_no_barrier_is_emitted_inside_the_per_element_wrap():
    msl = _compile().asm["msl"]
    assert not barrier_inside_wrap(msl), (
        "a threadgroup_barrier inside `for (_loop_e = lid; ...)` is undefined")


@requires_triton
@pytest.mark.xfail(strict=True, reason="the nesting is not served yet.")
def test_the_phases_are_emitted_fill_then_barrier_then_dot():
    msl = _compile().asm["msl"]
    order = phase_order(msl)
    assert order, "no phases were emitted at all"
    first_fill = order.index("fill")
    assert "barrier" in order[first_fill:], "no barrier after the first fill"
    first_barrier = order.index("barrier", first_fill)
    assert "dot" in order[first_barrier:], (
        f"the dot phase must follow the barrier that follows the fill; "
        f"order was {order}")
