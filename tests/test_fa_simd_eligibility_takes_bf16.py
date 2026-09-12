"""bf16 routes to the FA template that was built to take it, and computes.

Eighth wall on whisper, and the same class as the gate's mulf branch: layers
disagreeing about one dtype. The routing condition allows
`out_dtype in ("f32", "f16", "bf16")`. The dtype gate behind the routing says,
in its own words, why bf16 is safe there: "the tiled template promotes every
load to float, keeps the whole online softmax in fp32, and casts once on the
store, so bf16 there is a container and not a precision". The small-tile
eligibility accepts bf16. In the middle, `_simd_fa_eligible` said
`("f32", "f16")` -- while its own docstring discusses bf16 for head_dim>128.

Measured 2026-09-12, isolated through the real wrapper with the routing probe:

    (1500,1500) bf16 hd64: detection resolves {block 32x32, hd 64, bf16},
                           simd_eligible=False, small_tile=False
                           -> falls to the bf16 refusal.

One layer short of the template on every count but the dtype list.

The qk cap moves with the ELEMENT SIZE, not with the dtype's name: the Q
staging is BM*qk*elem bytes of threadgroup memory, and bf16 is two bytes like
fp16, so its cap is 192 -- the fp32 cap of 128 exists because fp32 is four.

Correctness is not inferred from the gate's comment: the wrapper-level test in
the engine repository compares the routed bf16 output against ATen. Here the
eligibility itself is pinned, both ways.

Runnable: python3 -m pytest tests/test_fa_simd_eligibility_takes_bf16.py -v
"""
from __future__ import annotations

import pytest

try:
    import triton_msl  # noqa: F401

    HAS = True
except Exception:                                     # pragma: no cover
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="triton_msl is not importable")


def _eligible(**overrides):
    from triton_msl.codegen.generic_lowerer import _simd_fa_eligible

    info = {
        "head_dim": 64, "out_dtype": "bf16",
        "block_m": 32, "block_n": 32,
        "strides": {r: ["z", "h", "m", "c1"] for r in ("q", "k", "v", "o")},
    }
    info.update(overrides)
    return _simd_fa_eligible(info)


@requires
def test_bf16_at_the_validated_shape_is_eligible():
    assert _eligible() is True, (
        "a cleanly-detected 32x32 hd64 bf16 kernel must route to the template "
        "the dtype gate says takes bf16; leaving it out sends every bf16 "
        "checkpoint to a refusal, and every weight in a modern checkpoint is "
        "bf16")


@requires
def test_the_qk_cap_follows_the_element_size():
    """bf16 is two bytes: its contraction cap is fp16's 192, not fp32's 128.

    The cap only bites when the OUTPUT width is a supported one while the
    contraction is wide -- the asymmetric MLA shape, qk=192 v=128 -- so
    `v_head_dim` is set explicitly here. A first draft omitted it, every
    variant then failed the vd-in-(64,128) rule before reaching the cap, and
    the equality it asserted held between two vacuous Falses.
    """
    mla = {"v_head_dim": 128, "head_dim": 192}
    f16_192 = _eligible(out_dtype="f16", **mla)
    assert f16_192 is True, (
        "the fixture must actually reach the cap: fp16 at qk=192 vd=128 is "
        "the validated MLA shape, and if this is False the two assertions "
        "below compare nothing")
    assert _eligible(out_dtype="bf16", **mla) == f16_192, (
        "bf16 and fp16 are the same element size; their caps must agree")
    assert _eligible(out_dtype="f32", **mla) is False, (
        "fp32 at 192 overflows the 32KB Q staging; the cap is about bytes, "
        "not about the dtype's name")


@requires
def test_what_was_ineligible_stays_ineligible():
    """Both directions: widening one dtype must not loosen anything else."""
    assert _eligible(out_dtype="i8") is False
    assert _eligible(block_m=16) is False, "the simd template is 32x32"
    assert _eligible(strides={r: ["z", "h", "m", "s"] for r in
                             ("q", "k", "v", "o")}) is False, (
        "non-contiguous stays with the scalar template")
    assert _eligible(head_dim=48) is False, "qk must be a multiple of 8, >= 64"
