"""A per-element ``scf.for`` region in a kernel whose tile exceeds the
threadgroup must cover the whole tile.

``GenericLowerer._is_scalar_op`` classes an ``scf.for`` as a scalar op (it
yields no tensor value), so a data-parallel region is hoisted out of the
multipass phase wrap loop and lowered with ``_needs_wrapping`` False — every
tile index in the region resolves to plain ``lid``, one element per thread.
When the tile is wider than the threadgroup that covers only ``lid`` of each
tile-stride: each load reads a fraction of its tile and each store writes a
fraction of its tile.

The store guard refused this case loudly (so it was never silently wrong at the
store), but its advice — launch with ``num_warps = BLOCK/32`` — is unreachable
whenever BLOCK > 1024: 4096/32 = 128 warps = 4096 threads, four times Metal's
1024-thread threadgroup limit. The loads in the same region had no guard.

The region is now emitted inside the same per-element loop the multipass path
uses for its non-reduce phases.
"""

import pytest
import torch
import triton
import triton.language as tl

HAS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
requires = pytest.mark.skipif(not HAS, reason="MPS + compile_shader needed")


@triton.jit
def _group_norm_two_pass(
    X, Y, W, B, Mean, Rstd,
    group_size, C, HW, num_groups, eps,
    scale_by_weight: tl.constexpr,
    add_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Two-pass group norm: a reduce loop, then a pure elementwise loop.

    Pass 1 makes the kernel take the multipass-reduce path (BLOCK 4096 > 1024,
    so the dispatch is capped to 1024 threads); pass 2 is the per-element
    ``scf.for`` region that has to cover all 4096.
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_groups
    group_idx = pid % num_groups

    chan_start = group_idx * group_size
    base_offset = batch_idx * C * HW + chan_start * HW
    hidden = group_size * HW

    block_range = tl.arange(0, BLOCK_SIZE)

    s = tl.zeros((), dtype=tl.float32)
    ss = tl.zeros((), dtype=tl.float32)
    for off in tl.range(0, hidden, BLOCK_SIZE):
        idx = off + block_range
        mask = idx < hidden
        x = tl.load(X + base_offset + idx, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(x)
        ss += tl.sum(x * x)

    inv_n = 1.0 / hidden
    mean = s * inv_n
    rstd = 1.0 / tl.sqrt(ss * inv_n - mean * mean + eps)

    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)

    for off in tl.range(0, hidden, BLOCK_SIZE):
        idx = off + block_range
        mask = idx < hidden
        x = tl.load(X + base_offset + idx, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        if scale_by_weight or add_bias:
            chan_in_group = idx // HW
            chan_global = chan_start + chan_in_group
            chan_mask = mask & (chan_in_group < group_size)
            if scale_by_weight:
                w = tl.load(W + chan_global, mask=chan_mask, other=1.0).to(tl.float32)
                x_hat = x_hat * w
            if add_bias:
                b = tl.load(B + chan_global, mask=chan_mask, other=0.0).to(tl.float32)
                x_hat = x_hat + b
        tl.store(Y + base_offset + idx, x_hat, mask=mask)


def _run(N, C, HW, num_groups):
    group_size = C // num_groups
    hidden = group_size * HW
    block_size = min(16384, triton.next_power_of_2(min(hidden, 16384)))

    torch.manual_seed(0)
    x = torch.randn(N, C, HW, device="mps", dtype=torch.float16)
    w = torch.randn(C, device="mps", dtype=torch.float16)
    b = torch.randn(C, device="mps", dtype=torch.float16)
    # NaN-filled so an element the kernel never writes is detectable, not just
    # numerically off.
    y = torch.full((N, C, HW), float("nan"), device="mps", dtype=torch.float16)
    mean = torch.zeros(N * num_groups, device="mps", dtype=torch.float32)
    rstd = torch.zeros(N * num_groups, device="mps", dtype=torch.float32)

    _group_norm_two_pass[(N * num_groups,)](
        x, y, w, b, mean, rstd,
        group_size, C, HW, num_groups, 1e-5,
        scale_by_weight=True, add_bias=True, BLOCK_SIZE=block_size,
    )
    torch.mps.synchronize()

    xf = x.to(torch.float32).reshape(N, num_groups, group_size * HW)
    m = xf.mean(dim=-1, keepdim=True)
    v = xf.var(dim=-1, unbiased=False, keepdim=True)
    ref = ((xf - m) / torch.sqrt(v + 1e-5)).reshape(N, C, HW)
    ref = ref * w.to(torch.float32)[None, :, None] + b.to(torch.float32)[None, :, None]
    return y, ref.to(torch.float16), block_size


@requires
def test_wide_per_element_region_covers_whole_tile():
    """hidden 2056 -> BLOCK 4096 > the 1024-thread threadgroup."""
    y, ref, block_size = _run(N=1, C=64, HW=257, num_groups=8)
    assert block_size == 4096, f"expected the wide tile, got BLOCK={block_size}"
    assert not torch.isnan(y).any(), (
        f"{int(torch.isnan(y).sum())} of {y.numel()} elements were never "
        f"written: the region covered only part of its tile"
    )
    torch.testing.assert_close(y.to(torch.float32), ref.to(torch.float32),
                               rtol=2e-3, atol=2e-2)


@requires
def test_narrow_region_unchanged():
    """hidden 132 -> BLOCK 256 <= the threadgroup: the path that already worked."""
    y, ref, block_size = _run(N=1, C=32, HW=33, num_groups=8)
    assert block_size <= 1024, f"expected a narrow tile, got BLOCK={block_size}"
    assert not torch.isnan(y).any()
    torch.testing.assert_close(y.to(torch.float32), ref.to(torch.float32),
                               rtol=2e-3, atol=2e-2)
