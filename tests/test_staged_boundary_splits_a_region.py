"""The phase splitter finds the staged-dot boundary, inside a region, and
stays byte-identical for every kernel that has none.

A `tt.dot` whose operands are staged through threadgroup memory is a phase
boundary for the same reason a reduce is: the staging is a cooperative loop
over all threads behind a barrier, and a barrier inside the per-element wrap
loop is undefined. The boundary is the `ttg.local_alloc` that feeds the dot,
not the dot: an unstaged dot is an ordinary in-loop op, and splitting on it
would break every kernel that has one.

Two things have to hold at once and only the pair is useful:

  * the split must FIRE on a region that carries the staging, or the nesting
    cannot be served;
  * it must not fire anywhere else, or every existing kernel changes shape.

Measured on the captured IR: the innermost of three nested loops holds 78 ops
and splits into five phases with the boundary, one without.

Runnable: python3 -m pytest tests/test_staged_boundary_splits_a_region.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

try:
    from triton.compiler.compiler import compile as triton_compile
    from triton.backends.compiler import GPUTarget
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    HAS = True
except Exception:                                     # pragma: no cover
    HAS = False

requires_triton = pytest.mark.skipif(not HAS, reason="triton + triton_msl needed")

FIXTURE = str(Path(__file__).parent / "golden" / "staged_dot_in_nested_loops.ttgir")


def _innermost_loop(ops):
    for op in ops:
        if op.op == "scf.for" and op.region_ops:
            return _innermost_loop(op.region_ops) or op
    return None


@pytest.fixture(autouse=True, scope="module")
def _hermetic_caches(tmp_path_factory):
    """Compile into caches this test owns.

    A compiled kernel is stashed on disk, and a stash written by an earlier
    run makes `make_msl` return without calling the lowerer at all -- so the
    test observes the cache, not the code. That has produced a false zero five
    times in this work already, twice in an instrument written to stop it, and
    a test is the last place it may happen: its result must not depend on what
    a previous run left behind.
    """
    import os
    d = tmp_path_factory.mktemp("caches")
    keys = {"TRITON_CACHE_DIR": str(d / "triton"),
            "TRITON_MSL_CACHE_DIR": str(d / "msl")}
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update(keys)
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def lowerer():
    """The lowerer instance that refused, with its parsed graph.

    Taken from the real compilation rather than rebuilt: a graph assembled by
    hand would be a second implementation of the parser, and the question is
    what the parser actually produces.
    """
    if not HAS:
        pytest.skip("triton needed")
    seen = []
    original = GenericLowerer.lower

    def spy(self, *a, **k):
        seen.append(self)
        return original(self, *a, **k)

    GenericLowerer.lower = spy
    try:
        try:
            triton_compile(FIXTURE, target=GPUTarget("metal", "apple-m4", 32))
        except Exception:
            pass                     # the refusal is the point; the graph is built
    finally:
        GenericLowerer.lower = original
    for low in seen:
        if getattr(low, "graph", None) and _innermost_loop(low.graph.ops):
            return low
    pytest.fail("no lowerer with a nested scf.for was reached by the fixture")


@requires_triton
def test_the_fixture_really_has_a_staged_dot_in_nested_loops(lowerer):
    """Without this the two tests below split an empty body and pass."""
    loop = _innermost_loop(lowerer.graph.ops)
    body = [o.op for o in loop.region_ops]
    assert "ttg.local_alloc" in body, f"no staging in the innermost body: {body[:8]}"
    assert "tt.dot" in body, f"no dot in the innermost body: {body[:8]}"


@requires_triton
def test_the_boundary_splits_the_innermost_body(lowerer):
    loop = _innermost_loop(lowerer.graph.ops)
    phases = lowerer._split_ops_by_reductions(list(loop.region_ops), staged_dot=True)
    boundaries = [ops for ops, is_b in phases if is_b]
    assert len(phases) > 1, (
        "the body must decompose into phases; one phase is the shape that "
        "cannot be served, since the wrap would enclose the staging again")
    assert boundaries, "no boundary phase was isolated"
    assert all(o.op in GenericLowerer._STAGED_BOUNDARY
               for ops in boundaries for o in ops), (
        "a boundary phase must hold only the staging op it is named for")


@requires_triton
def test_without_the_flag_the_body_is_one_phase(lowerer):
    """The widening is inert for every existing caller.

    `_split_ops_by_reductions()` is called from the top-level multipass path
    with no arguments, and this pins that the new boundary cannot reach it by
    accident: the same body that splits five ways above stays whole here.
    """
    loop = _innermost_loop(lowerer.graph.ops)
    phases = lowerer._split_ops_by_reductions(list(loop.region_ops))
    assert len(phases) == 1 and not phases[0][1], (
        f"without staged_dot the body must stay one non-boundary phase; "
        f"got {len(phases)}")


# ── which shapes the nesting may be attempted on, and which it must decline ─


class _Op:
    """A duck-typed op. The predicate reads four attributes and nothing else,
    so a fake carries the whole contract and the negative cases need no IR."""

    def __init__(self, op, region_ops=None, else_ops=None):
        self.op = op
        self.region_ops = region_ops or []
        self.else_ops = else_ops or []


def _predicate(ops):
    if not HAS:
        pytest.skip("triton_msl needed")
    return GenericLowerer.staged_dot_loop_nest(GenericLowerer, ops)


@requires_triton
def test_the_real_shape_is_recognised(lowerer):
    """The fixture is the shape this is for; if it is not recognised the
    whole nesting path is unreachable and every negative case below is
    vacuously satisfied."""
    assert lowerer.staged_dot_loop_nest(lowerer.graph.ops) is not None


@requires_triton
def test_a_loop_staging_a_dot_is_recognised():
    nest = _Op("scf.for", [_Op("tt.load"), _Op("ttg.local_alloc"), _Op("tt.dot")])
    assert _predicate([nest]) is nest, (
        "the returned value is the NEST, not a boolean: the caller needs the "
        "loop to emit phases inside")


@requires_triton
def test_a_loop_without_staging_is_not_the_shape():
    assert _predicate([_Op("scf.for", [_Op("tt.load"), _Op("tt.dot")])]) is None, (
        "an unstaged dot in a loop needs no cooperative phase and no barrier; "
        "taking the nesting path for it would restructure a kernel that is "
        "already correct")


@requires_triton
def test_a_barrier_outside_the_nest_declines():
    """The case that turns a refusal into invalid MSL if it is got wrong.

    A top-level reduce needs the multipass wrap it already has. Moving the
    wrap inside the loops would leave that reduce outside every per-element
    loop, uncovered -- and it would not announce itself, because the kernel
    still compiles.
    """
    nest = _Op("scf.for", [_Op("ttg.local_alloc"), _Op("tt.dot")])
    assert _predicate([_Op("tt.reduce"), nest]) is None
    assert _predicate([nest, _Op("tt.scan")]) is None
    assert _predicate([nest, _Op("ttg.local_alloc")]) is None, (
        "staging outside the nest is staging the nesting cannot place")


@requires_triton
def test_staging_nested_two_loops_deep_is_still_found():
    """The captured shape is three loops deep, not one."""
    inner = _Op("scf.for", [_Op("ttg.local_alloc"), _Op("tt.dot")])
    mid = _Op("scf.for", [inner])
    outer = _Op("scf.for", [mid])
    assert _predicate([outer]) is outer, (
        "the OUTERMOST loop of the nest is what the phases must be emitted "
        "inside; returning the innermost would leave the outer loops wrapped")
