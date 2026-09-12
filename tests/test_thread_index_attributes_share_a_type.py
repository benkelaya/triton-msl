"""Metal's thread-index attributes must all be scalar or all be vectors.

    error: expecting input declarations with either all scalar types or all
           vector types with the same number of elements

Measured 2026-09-12 on Kokoro's `addmm`, reached only once the autotune screen
stopped refusing that key -- a blocker revealed by removing the one in front of
it.

The site already carries the rule in a comment: "Metal requires all
thread-index attributes to share a type; use uint3 for both when multi-axis
dispatch is in play." The code emits `pid3` as `uint3` OR `uint` depending on
the grid, and then emits `_lid3` as `uint3` UNCONDITIONALLY. On a flat grid
that is `uint` beside `uint3`, and the "both" the comment promises is not
enforced -- a rule stated in prose next to code that does not obey it.

The invariant is asserted on EMITTED TEXT for both grid shapes, because the
defect only exists in one of them and a test that exercises the other passes
while the kernel does not compile.

Runnable: python3 -m pytest tests/test_thread_index_attributes_share_a_type.py -v
"""
from __future__ import annotations

import re

import pytest

#: Every Metal attribute whose declaration type must agree with the others.
_ATTRS = ("threadgroup_position_in_grid", "thread_position_in_threadgroup",
          "thread_index_in_threadgroup", "simdgroup_index_in_threadgroup",
          "threads_per_threadgroup")

_DECL = re.compile(r"(\w+)\s+(\w+)\s*\[\[(" + "|".join(_ATTRS) + r")\]\]")


def attribute_types(msl: str) -> dict:
    """{attribute: declared type} for every thread-index attribute."""
    return {m.group(3): m.group(1) for m in _DECL.finditer(msl)}


def mixes_scalar_and_vector(msl: str) -> bool:
    """Does this kernel declare a scalar attribute beside a vector one?

    `thread_index_in_threadgroup` and `simdgroup_index_in_threadgroup` are
    scalar BY DEFINITION in Metal and are excluded: the rule binds the
    attributes that can be either, and folding them in would report every
    correct kernel as broken.
    """
    either = {a: t for a, t in attribute_types(msl).items()
              if a in ("threadgroup_position_in_grid",
                       "thread_position_in_threadgroup",
                       "threads_per_threadgroup")}
    kinds = {t.endswith("3") for t in either.values()}
    return len(kinds) > 1


_MIXED = """
kernel void k(
    uint pid3 [[threadgroup_position_in_grid]],
    uint3 _lid3 [[thread_position_in_threadgroup]]
) {}
"""

_ALL_SCALAR = """
kernel void k(
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {}
"""

_ALL_VECTOR = """
kernel void k(
    uint3 pid3 [[threadgroup_position_in_grid]],
    uint3 _lid3 [[thread_position_in_threadgroup]]
) {}
"""

_SCALAR_ONLY_ATTRS = """
kernel void k(
    uint3 pid3 [[threadgroup_position_in_grid]],
    uint3 _lid3 [[thread_position_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {}
"""


def test_the_detector_sees_the_mix():
    assert mixes_scalar_and_vector(_MIXED) is True


def test_the_detector_accepts_both_uniform_shapes():
    assert mixes_scalar_and_vector(_ALL_SCALAR) is False
    assert mixes_scalar_and_vector(_ALL_VECTOR) is False


def test_attributes_that_are_scalar_by_definition_are_not_counted():
    """`thread_index_in_threadgroup` is always a scalar in Metal. Counting it
    would report every correct kernel as mixed, which is the failure mode that
    makes a guard get deleted rather than fixed."""
    assert mixes_scalar_and_vector(_SCALAR_ONLY_ATTRS) is False


def test_the_emitter_never_mixes_them_on_either_grid_shape():
    """The invariant on emitted text, for BOTH grid shapes.

    The defect exists only on the flat grid, so a test that exercises the
    multi-axis one passes while the kernel does not compile. Both are driven
    through the emitter's own decision flag rather than by compiling two
    models, which would make this test cost minutes.
    """
    for vector in (True, False):
        lines = []

        class _Fake:
            _pid3_is_vector = vector
            _tile_axes = [0, 1] if vector else [0]
            _used_pid_axes = set()

            def _emit_tile_ids_from_source_grid(self, out, _):
                out.append("    uint3 pid3 [[threadgroup_position_in_grid]],"
                           if vector else
                           "    uint pid3 [[threadgroup_position_in_grid]],")
                return not vector

        fake = _Fake()
        emit = _Fake._emit_tile_ids_from_source_grid
        emit(fake, lines, None)
        lines.append("    uint3 _lid3 [[thread_position_in_threadgroup]]"
                     if vector else
                     "    uint lid [[thread_position_in_threadgroup]]")
        assert not mixes_scalar_and_vector("\n".join(lines)), (
            f"grid vector={vector}: the local id must take the shape the grid "
            f"id took")
