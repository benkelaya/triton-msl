"""A range that reaches its broadcast through ``//`` or ``%`` keeps its axis.

For a 2-D tile the lowerer maps thread ``lid`` to ``(lid / N, lid % N)``.
Which of the two a ``tt.make_range`` gets is read from the ``tt.expand_dims``
that broadcasts it, by walking back from the expand_dims operand to the range.

That walk crossed ``arith.addi`` and ``arith.muli`` and nothing else, so a flat
index decomposed with ``//`` and ``%`` — the im2col idiom — lost its axis and
fell back to the COLUMN index. Every thread of a row then addressed the same
element, and the kernel returned a plausible array that was wrong.

Measured 2026-09-08 on a real convolution: 9.894e-01 relative error against an
fp64 oracle on both autotune configs that compile, output constant along the
width axis. Nothing refused. It surfaced only because the two configs were
wrong DIFFERENTLY and the autotune consensus screen would not choose.

The kernel below is exact in integers, so there is no tolerance to hide in.

A second test was written alongside this one and then deleted: it asserted
that the 32 output rows were all distinct, on the reasoning that a collapsed
row index makes threads address the same row. Run against the unfixed lowerer
it PASSED — the collapsed index leaves most rows zero and a handful written,
which is still 32 distinct rows. A test that cannot fail on the defect it
names is worse than no test, so it is not here. The exact-equality check
below fails on the unfixed lowerer with rows [0..7] differing, which is the
whole point.
"""

import pytest
import torch
import triton
import triton.language as tl

HAS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
requires = pytest.mark.skipif(not HAS, reason="MPS + compile_shader needed")

H, W, K = 8, 4, 32          # H*W = 32 rows, K = 32 columns -> 1024 threads


@triton.jit
def _hw_transpose_block(
    OUT, SRC,
    H_: tl.constexpr, W_: tl.constexpr,
    BLOCK_BHW: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Copy a (BLOCK_BHW x BLOCK_K) block, transposing h and w on the way.

    ``bhw`` is the tile's ROW. It reaches its ``[:, None]`` broadcast ONLY
    through ``//`` and ``%`` — never through a bare add or multiply — which is
    exactly the chain the walk used to stop at. The h/w transpose means there
    is no algebraic identity for the compiler to fold the decomposition back
    into, so the ``divsi``/``remsi`` really are on the path.
    """
    bhw = tl.arange(0, BLOCK_BHW)
    k = tl.arange(0, BLOCK_K)

    h = bhw // W_
    w = bhw % W_

    src = SRC + (h * (W_ * BLOCK_K))[:, None] + (w * BLOCK_K)[:, None] + k[None, :]
    dst = OUT + (w * (H_ * BLOCK_K))[:, None] + (h * BLOCK_K)[:, None] + k[None, :]
    tl.store(dst, tl.load(src))


@requires
def test_row_range_through_div_and_mod_is_not_the_column():
    src = torch.arange(H * W * K, dtype=torch.float32, device="mps").reshape(H * W, K)
    out = torch.zeros_like(src)

    _hw_transpose_block[(1,)](
        out, src, H_=H, W_=W, BLOCK_BHW=H * W, BLOCK_K=K,
    )

    want = src.reshape(H, W, K).permute(1, 0, 2).reshape(W * H, K)
    got = out.cpu()
    assert torch.equal(got, want.cpu()), (
        "the row range collapsed onto the column index: rows "
        f"{sorted(set((got != want.cpu()).any(dim=1).nonzero().flatten().tolist()))[:8]} "
        "differ"
    )
