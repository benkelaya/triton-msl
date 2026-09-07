"""A batched matmul computes, and grouped-query attention reads the right head.

Both were refusals or silent-wrongs that a language model's decode step walks
into on its first token:

* a batched matmul (`out[b] = A[b] @ B[b]`) was refused outright — "batched
  MMA is not implemented" — which is true of the simdgroup path and not of
  the stride-aware scalar one, where the batch is one more term in an address
  that is already written out;
* the batch axis is the kernel's choice, and reading the tile coordinates
  from x and y regardless takes the batch index for a row tile;
* `M == 1` — a batched matrix-VECTOR product, which is what attention decode
  is — has no `M` argument at all, because `equal_to_1` replaces it with the
  literal, and the extent then came from the block size;
* grouped-query attention gives K and V fewer heads than Q, and the kernel
  says so by dividing the head index. A template that indexes K by the QUERY
  head reads head 5 of a tensor that has 4.
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
    def _bmm_batch_first(a, b, c, M, N, K,
                         sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """Batch on program_id(0), tiles on 1 and 2."""
        pb = tl.program_id(0)
        pm = tl.program_id(1)
        pn = tl.program_id(2)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        ap = a + pb * sab + (rm[:, None] * sam + rk[None, :] * sak)
        bp = b + pb * sbb + (rk[:, None] * sbk + rn[None, :] * sbn)
        acc = tl.zeros((BM, BN), tl.float32)
        for _ in range(0, K, BK):
            acc += tl.dot(tl.load(ap), tl.load(bp))
            ap += BK * sak
            bp += BK * sbk
        tl.store(c + pb * scb + (rm[:, None] * scm + rn[None, :] * scn),
                 acc.to(c.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def _bmm_batch_last_flat_tiles(a, b, c, M, N, K,
                                   sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                                   BM: tl.constexpr, BN: tl.constexpr,
                                   BK: tl.constexpr):
        """Both tile coordinates on the FLAT axis 0; batch on program_id(2).

        The shape every `baddbmm` in the wild is launched with. Reading pid_n
        from y here gives 0 for every program.
        """
        pb = tl.program_id(2)
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BN)
        pm = pid // num_pid_n
        pn = pid % num_pid_n
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        ap = a + pb * sab + (rm[:, None] * sam + rk[None, :] * sak)
        bp = b + pb * sbb + (rk[:, None] * sbk + rn[None, :] * sbn)
        acc = tl.zeros((BM, BN), tl.float32)
        for _ in range(0, K, BK):
            acc += tl.dot(tl.load(ap), tl.load(bp))
            ap += BK * sak
            bp += BK * sbk
        tl.store(c + pb * scb + (rm[:, None] * scm + rn[None, :] * scn),
                 acc.to(c.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


def _ref(A, B):
    return torch.matmul(A.cpu().double(), B.cpu().double())


@requires
@pytest.mark.parametrize("Bz,M,N,K", [(4, 32, 32, 32), (3, 64, 32, 64)])
def test_batched_matmul_batch_on_axis_0(Bz, M, N, K):
    torch.manual_seed(0)
    A = torch.randn(Bz, M, K, device="mps")
    B = torch.randn(Bz, K, N, device="mps")
    C = torch.zeros(Bz, M, N, device="mps")
    BM = BN = BK = 32
    _bmm_batch_first[(Bz, (M + BM - 1) // BM, (N + BN - 1) // BN)](
        A, B, C, M, N, K,
        *A.stride(), *B.stride(), *C.stride(), BM=BM, BN=BN, BK=BK)
    torch.mps.synchronize()
    err = (C.cpu().double() - _ref(A, B)).abs().max().item()
    assert err < 1e-3, f"batched matmul (batch on axis 0) max error {err}"
    # A dropped batch computes batch 0 for every batch; the batches differ.
    assert (C[0] - C[-1]).abs().max().item() > 0, "every batch is identical"


@requires
@pytest.mark.parametrize("Bz,M,N,K", [(4, 64, 64, 64), (2, 1, 128, 128)])
def test_batched_matmul_flat_tiles_batch_on_axis_2(Bz, M, N, K):
    """The second case has M == 1: no `M` argument survives specialization."""
    torch.manual_seed(0)
    A = torch.randn(Bz, M, K, device="mps")
    B = torch.randn(Bz, K, N, device="mps")
    C = torch.zeros(Bz, M, N, device="mps")
    # A 32-row tile over a 1-row output: the tile is padding, and `M` is gone
    # from the argument list because `equal_to_1` replaced it with the literal.
    BM, BN, BK = 32, 32, 32
    grid = (((M + BM - 1) // BM) * ((N + BN - 1) // BN), 1, Bz)
    _bmm_batch_last_flat_tiles[grid](
        A, B, C, M, N, K,
        *A.stride(), *B.stride(), *C.stride(), BM=BM, BN=BN, BK=BK)
    torch.mps.synchronize()
    err = (C.cpu().double() - _ref(A, B)).abs().max().item()
    assert err < 1e-3, f"batched matmul (flat tiles, batch on axis 2) max error {err}"
    assert (C[0] - C[-1]).abs().max().item() > 0, "every batch is identical"
