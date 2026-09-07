"""In-loop 2-D axis reduce (GEMV-via-tl.sum): correct.

A 2-D→1-D axis reduce (`tl.sum(x, axis=1)`) has two correct read-back
layouts. The row-broadcast one (thread ``lid`` holds row ``lid/N``) is what a
2-D consumer needs — `x - tl.max(x, 1)[:, None]`. One row per thread is what
a loop-carried accumulator and a 1-D store need.

Emitting the first where the second was wanted collapsed every output row onto
the first, so it was REFUSED until 2026-09-07, when the read-back learned to
take the layout its consumer asks for. These tests assert the answers.

A result that needs BOTH at once still refuses: no single read serves them.
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal

    from triton_msl.errors import MetalNonRecoverableError

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + torch + triton needed")

if HAS:

    @triton.jit
    def _gemv_loop(x_ptr, w_ptr, o_ptr, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for _ in range(0, K, BK):
            x = tl.load(x_ptr + offs_k)
            w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk)
            acc += tl.sum(x[None, :] * w, axis=1)
            offs_k += BK
        tl.store(o_ptr + offs_n, acc)

    @triton.jit
    def _gemv_single(x_ptr, w_ptr, o_ptr, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        x = tl.load(x_ptr + offs_k)
        w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk)
        tl.store(o_ptr + offs_n, tl.sum(x[None, :] * w, axis=1))

    @triton.jit
    def _gemv_advancing_pointer(x_ptr, w_ptr, o_ptr, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
        """The GEMV as every reference writes it: the operand POINTERS are
        carried across the loop and advanced, rather than the offsets.

        The value going round the loop is an address. Carried as a number it
        emitted `float iter = w_ptr[...]` and then subscripted that float,
        which does not compile — so this shape refused, the vocabulary
        projection of a language model with it."""
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        x_ptrs = x_ptr + offs_k
        w_ptrs = w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk
        for _ in range(0, K, BK):
            x = tl.load(x_ptrs)
            w = tl.load(w_ptrs)
            acc += tl.sum(x[None, :] * w, axis=1)
            x_ptrs += BK
            w_ptrs += BK * swk
        tl.store(o_ptr + offs_n, acc)

    @triton.jit
    def _gemv_both_layouts(x_ptr, w_ptr, o_ptr, o2_ptr, K, swn, swk,
                           BN: tl.constexpr, BK: tl.constexpr):
        """The loop-carried reduce result used BOTH ways: stored 1-D and
        broadcast back to 2-D. One value cannot be read both ways."""
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for _ in range(0, K, BK):
            x = tl.load(x_ptr + offs_k)
            w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk)
            acc += tl.sum(x[None, :] * w, axis=1)
            offs_k += BK
        tl.store(o_ptr + offs_n, acc)
        w2 = tl.load(w_ptr + offs_n[:, None] * swn + tl.arange(0, BK)[None, :] * swk)
        tl.store(o2_ptr + offs_n[:, None] * BK + tl.arange(0, BK)[None, :],
                 w2 * acc[:, None])

    @triton.jit
    def _gemv_single_iterarg(x_ptr, w_ptr, o_ptr, N, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
        # A plain fp32 GEMV with a SINGLE loop iter_arg (`acc`) and a
        # `range(0, K, BK)` induction var (no manual offs_k += BK). A single-result
        # scf.for has result_ids == None, which an earlier version of the guard failed
        # to cross — the same silent-wrong slipped through. It must refuse. (No dequant
        # / no sitofp, so it is NOT routed to the int8 GEMV kernel — it hits the guard.)
        pid = tl.program_id(0)
        offs_n = pid * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k in range(0, K, BK):
            x = tl.load(x_ptr + offs_k + k)
            w = tl.load(w_ptr + offs_n[:, None] * swn + (offs_k[None, :] + k) * swk)
            acc += tl.sum(x[None, :] * w, axis=1)
        tl.store(o_ptr + offs_n, acc)


@requires
def test_inloop_2d_axis_reduce_is_correct():
    """K > BK forces the K-loop that carries the reduce result.

    This is the shape that used to collapse every output row onto row 0 — and
    the collapse is what the assertion catches: a wrong layout gives every
    element the same value, which `w @ x` is not.
    """
    torch.manual_seed(0)
    N, K, BK = 32, 64, 32
    x = torch.randn(K, device="mps")
    w = torch.randn(N, K, device="mps")
    o = torch.zeros(N, device="mps")
    _gemv_loop[(1,)](x, w, o, K, w.stride(0), w.stride(1), BN=N, BK=BK)
    torch.mps.synchronize()
    torch.testing.assert_close(o, w @ x, rtol=1e-3, atol=1e-3)
    assert o.unique().numel() > 1, "every output row is the same value"


@requires
def test_single_iterarg_loop_carried_reduce_is_correct():
    """One loop iter_arg — a single-result `scf.for`, whose ``result_ids`` is
    None — carried across the loop and stored 1-D. The decode GEMV's shape."""
    torch.manual_seed(0)
    N, K, BN, BK = 128, 256, 32, 32
    x = torch.randn(K, device="mps")
    w = torch.randn(N, K, device="mps")
    o = torch.zeros(N, device="mps")
    _gemv_single_iterarg[(triton.cdiv(N, BN),)](
        x, w, o, N, K, w.stride(0), w.stride(1), BN=BN, BK=BK)
    torch.mps.synchronize()
    torch.testing.assert_close(o, w @ x, rtol=1e-2, atol=1e-2)
    assert o.unique().numel() > 1, "every output row is the same value"


@requires
def test_single_tile_2d_axis_reduce_still_correct():
    # No K-loop (BK == K): the reduce result is consumed directly, correctly.
    torch.manual_seed(0)
    N, K = 32, 32
    x = torch.randn(K, device="mps")
    w = torch.randn(N, K, device="mps")
    o = torch.zeros(N, device="mps")
    _gemv_single[(1,)](x, w, o, w.stride(0), w.stride(1), BN=N, BK=K)
    torch.mps.synchronize()
    torch.testing.assert_close(o, w @ x, rtol=1e-3, atol=1e-3)


@requires
def test_loop_advanced_pointer_gemv_is_correct():
    """The operand pointers are advanced inside the loop, not the offsets."""
    torch.manual_seed(0)
    N, K, BK = 32, 128, 32
    x = torch.randn(K, device="mps")
    w = torch.randn(N, K, device="mps")
    o = torch.zeros(N, device="mps")
    _gemv_advancing_pointer[(1,)](x, w, o, K, w.stride(0), w.stride(1),
                                  BN=N, BK=BK)
    torch.mps.synchronize()
    torch.testing.assert_close(o, w @ x, rtol=1e-2, atol=1e-2)
    assert o.unique().numel() > 1, "every output row is the same value"


@requires
def test_reduce_needing_both_layouts_refuses():
    """Stored 1-D AND broadcast back to 2-D: the two consumers want different
    read-backs of the same value, so it refuses rather than serve one."""
    torch.manual_seed(0)
    N, K, BK = 32, 64, 32
    x = torch.randn(K, device="mps")
    w = torch.randn(N, K, device="mps")
    o = torch.zeros(N, device="mps")
    o2 = torch.zeros(N * BK, device="mps")
    with pytest.raises(MetalNonRecoverableError, match="BOTH stored 1-D"):
        _gemv_both_layouts[(1,)](x, w, o, o2, K, w.stride(0), w.stride(1),
                                 BN=N, BK=BK)


@requires
def test_two_d_reduce_wider_than_the_threadgroup_refuses_with_its_number():
    """A tile the threadgroup cannot cover names the number that has to change,
    rather than reaching the caller as "could not lower this kernel"."""
    torch.manual_seed(0)
    N, K, BK = 64, 128, 64          # 64 x 64 = 4096 > 1024
    x = torch.randn(K, device="mps")
    w = torch.randn(N, K, device="mps")
    o = torch.zeros(N, device="mps")
    with pytest.raises(MetalNonRecoverableError, match="exceeds the 1024-thread"):
        _gemv_loop[(1,)](x, w, o, K, w.stride(0), w.stride(1), BN=N, BK=BK)
