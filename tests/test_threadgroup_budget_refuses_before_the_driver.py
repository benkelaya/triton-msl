"""A kernel over the threadgroup budget refuses at codegen, not at the driver.

Measured 2026-09-12, after a refused autotune config stopped ending the sweep:
hat-s-x4 and swin2SR both reached further into their model and then died with

    AGXMetalG16X Code=3 "Threadgroup memory size (49152) exceeds the
    maximum threadgroup memory allowed (32768)"

raised where the pipeline state is created -- after the kernel compiled, after
its metallib was built. Three things follow from that layer being the wrong
one:

  * the whole run dies. A `MetalNonRecoverableError` raised at codegen is a
    refusal the autotune sweep already knows how to exclude, scoring that
    config `inf` and trying the next; a driver error is not, and nothing in
    the sweep catches it.
  * the cost is paid first. Reaching the driver means the config was lowered,
    emitted, compiled by `xcrun metal` and linked -- all of it thrown away for
    a budget that could be counted from the emitted text.
  * the message names a number and not a cause. Which buffers, and how big,
    is what a reader needs; `49152` alone sends them to the MSL by hand.

The tree already draws this line for the 1024-thread ceiling, in its own
words: "for an AUTOTUNED kernel it takes the whole screen down instead of
letting the tuner skip that config".

The budget is `metal_threadgroup_bytes()`, read from the backend's
`MetalOptions.max_threadgroup_memory` when it is loaded. Not a literal here:
a second copy of a device limit is a second thing to forget.

Runnable: python3 -m pytest tests/test_threadgroup_budget_refuses_before_the_driver.py -v
"""
from __future__ import annotations

import pytest

# The skip covers ONE thing: this tree not being importable at all. It must
# not cover the accounting being absent -- a module-level `except Exception`
# around both turned five red tests into five skips, and a skip is not a
# success. So the package import is guarded and the functions under test are
# imported inside the tests, where a missing one fails.
try:
    import triton_msl  # noqa: F401

    HAS = True
except Exception:                                     # pragma: no cover
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="triton_msl is not importable")


def _budget():
    from triton_msl.codegen._lowerer_reduce import metal_threadgroup_bytes
    return metal_threadgroup_bytes()


def _used(msl):
    from triton_msl.codegen._lowerer_helpers import threadgroup_bytes_used
    return threadgroup_bytes_used(msl)


_UNDER = """
kernel void k() {
    threadgroup float a[2048];
    threadgroup float b[1024];
}
"""

_OVER = """
kernel void k() {
    threadgroup float a[4096];
    threadgroup float b[4096];
    threadgroup float c[4096];
}
"""

_MIXED_TYPES = """
kernel void k() {
    threadgroup half h[1024];
    threadgroup uint u[512];
}
"""


@requires
def test_the_budget_comes_from_the_backend_not_from_a_literal():
    assert _budget() == 32768, (
        "this machine's declared budget changed; the accounting below is "
        "measured against whatever the backend declares, but the fixtures "
        "were sized for 32768 and must be resized with it")


@requires
def test_the_accounting_sums_every_declaration_with_its_element_size():
    assert _used(_UNDER) == (2048 + 1024) * 4
    assert _used(_OVER) == 4096 * 3 * 4
    assert _used(_MIXED_TYPES) == 1024 * 2 + 512 * 4, (
        "a half is two bytes and a uint is four; counting elements instead of "
        "bytes would pass a kernel that does not fit and refuse one that does")


@requires
def test_a_kernel_with_no_threadgroup_memory_counts_zero():
    assert _used("kernel void k() { int x = 0; }") == 0


@requires
def test_over_budget_is_refused_and_the_refusal_names_the_buffers():
    from triton_msl.codegen._lowerer_helpers import refuse_over_threadgroup_budget
    from triton_msl.errors import MetalNonRecoverableError

    with pytest.raises(MetalNonRecoverableError) as exc:
        refuse_over_threadgroup_budget(_OVER)
    msg = str(exc.value)
    assert "49152" in msg and "32768" in msg, "both numbers must appear"
    for name in ("a", "b", "c"):
        assert f" {name}" in msg or f"{name}[" in msg, (
            f"the refusal must name buffer {name}: a total without its "
            f"parts sends the reader back to the MSL by hand")


@requires
def test_under_budget_passes():
    """Both directions. A check that refuses everything would also make every
    measurement above read as a success."""
    from triton_msl.codegen._lowerer_helpers import refuse_over_threadgroup_budget

    refuse_over_threadgroup_budget(_UNDER)          # must not raise
