"""FlashAttention where the query and key sequences are not the same length,
and where the softmax scale reaches the dot through a dtype cast.

Both cases are ordinary — the first is every token a language model emits
after the first, the second is what `q = (q * scale).to(q.dtype)` compiles to
on a fp16 kernel — and both were refused before the changes these tests
cover:

* the scale walk-back stopped at the `arith.truncf` a fp16 kernel inserts
  between the multiply and the dot, so the scale did not resolve, the kernel
  was not recognized as attention, and it refused further down the generic
  path for an unrelated reason (a 2-D axis reduce inside a loop);
* a query tile below 32 rows — what a hardware profile picks for
  `seqlen_q = 1` — was refused by a guard written for the generic per-thread
  lowering, which the tiled template does not share;
* the query length of a decode step is 1, which Triton's `equal_to_1`
  specialization replaces with a literal, and both the bound resolver and the
  store-mask triviality proof accepted only a scalar ARGUMENT.

The reference is computed in float64 and compared in the kernel's dtype.
"""

import pytest
import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

requires_triton = pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")

try:
    from triton_msl.errors import MetalNonRecoverableError
except Exception:  # pragma: no cover
    MetalNonRecoverableError = Exception


@triton.jit
def _fa_two_lengths(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, seqlen_q, seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """FA2 forward with independent query / key lengths.

    The scale is applied to Q and cast BACK to Q's dtype, which is what the
    reference implementations do and what puts an `arith.truncf` between the
    multiply and the dot on a fp16 kernel.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = (Q + off_z * stride_qz + off_h * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)

    qk_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
    q = (q * qk_scale).to(q.dtype)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, seqlen_k, BLOCK_N):
        k_ptrs = (K + off_z * stride_kz + off_h * stride_kh
                  + (start_n + offs_n)[:, None] * stride_kn
                  + offs_d[None, :] * stride_kk)
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)

        qk = tl.dot(q, tl.trans(k).to(q.dtype))

        if IS_CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptrs = (V + off_z * stride_vz + off_h * stride_vh
                  + (start_n + offs_n)[:, None] * stride_vn
                  + offs_d[None, :] * stride_vk)
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)

        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = (Out + off_z * stride_oz + off_h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)


def _oracle(q, k, v):
    """softmax(q k^T / sqrt(d)) v in float64, written out."""
    q64, k64, v64 = q.double(), k.double(), v.double()
    scores = torch.matmul(q64, k64.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    return torch.matmul(torch.softmax(scores, dim=-1), v64)


def _run(seqlen_q, seqlen_k, BLOCK_M, BLOCK_N, dtype, head_dim=64, H=2,
         causal=False):
    torch.manual_seed(7)
    Z = 1
    q = torch.randn(Z, H, seqlen_q, head_dim, dtype=torch.float32).to(dtype)
    k = torch.randn(Z, H, seqlen_k, head_dim, dtype=torch.float32).to(dtype)
    v = torch.randn(Z, H, seqlen_k, head_dim, dtype=torch.float32).to(dtype)
    out = torch.zeros_like(q)
    grid = ((seqlen_q + BLOCK_M - 1) // BLOCK_M, Z * H)
    _fa_two_lengths[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, seqlen_q, seqlen_k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim, IS_CAUSAL=causal,
    )
    return out, _oracle(q, k, v)


@requires_triton
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 4e-3), (torch.float32, 1e-5)])
def test_scale_through_a_dtype_cast_is_resolved(dtype, tol):
    """`(q * scale).to(q.dtype)` must still resolve the scale.

    On fp16 the cast is an `arith.truncf` sitting between the multiply and the
    dot. Walking past it is what tells the kernel apart from one with no scale
    at all; without it a fp16 attention refuses.
    """
    out, ref = _run(64, 64, 32, 32, dtype)
    assert torch.isfinite(out).all()
    assert (out.double() - ref).abs().max().item() < tol


@requires_triton
@pytest.mark.parametrize("BLOCK_M", [8, 16])
def test_decode_one_query_row_against_a_long_key_sequence(BLOCK_M):
    """seqlen_q = 1 against seqlen_k = 128 — a decode step.

    The query tile is under 32 rows and the query length is a literal, both of
    which used to refuse. Correctness is what is asserted: the tile is padding,
    not a shorter computation.
    """
    out, ref = _run(1, 128, BLOCK_M, 32, torch.float16)
    assert torch.isfinite(out).all()
    assert (out.double() - ref).abs().max().item() < 4e-3


@requires_triton
def test_query_shorter_than_keys_does_not_read_past_q():
    """A query tile wider than the query sequence must not run off the end.

    With one bound for both, the template guarded the query rows with the KEY
    length: 8 query rows against 128 keys read 128 rows of Q. The rows past
    the end are guarded, so the answer stays the oracle's.
    """
    out, ref = _run(8, 128, 16, 32, torch.float32)
    assert torch.isfinite(out).all()
    assert (out.double() - ref).abs().max().item() < 1e-5


@requires_triton
def test_causal_with_distinct_lengths_refuses():
    """A causal mask over different lengths has to say where the queries sit.

    Aligned to the start of the key sequence or to its end are both real
    conventions and they disagree; the IR does not say which. Refusing is the
    only answer that cannot be silently wrong.
    """
    with pytest.raises(MetalNonRecoverableError):
        _run(16, 128, 16, 32, torch.float32, causal=True)
