"""An absent `v_head_dim` must mean the same thing to every reader.

`v_head_dim` is the attention's OUTPUT width; `head_dim` is the QK
contraction width. They differ only for asymmetric attention (MLA /
DeepSeek: qk=192, v=128). Symmetric attention has them equal.

Five sites read the key. Four defaulted to `head_dim` — "absent means
symmetric". The fifth read `info.get("v_head_dim") != info["head_dim"]`
with no default, so an absent key was `None`, `None != head_dim` was True,
and that site alone read the same absence as "asymmetric" — routing a
symmetric kernel down the asymmetric path.

Only one producer sets the key today, so production never took the
contradictory branch. The upstream suite already builds a dict without it
(`test_fa_simdgroup_routing.py`), and a second producer would have found it.
These tests pin the answer rather than the five copies of a default.
"""
import pytest

from triton_msl.codegen.generic_lowerer import (
    _v_head_dim_of, _simd_fa_eligible, _fa_small_tile_eligible,
)


def _info(**over):
    """The shape the detector produces, minus whatever a test drops."""
    info = {
        "head_dim": 128,
        "block_m": 32,
        "block_n": 32,
        "out_dtype": "f32",
        "causal": False,
        "scale": 0.0883,
        "strides": {r: ["c1", "c1", 2, "c1"] for r in ("q", "k", "v", "o")},
    }
    info.update(over)
    return info


# ------------------------------------------------------------ the accessor

def test_absent_means_symmetric():
    assert _v_head_dim_of({"head_dim": 128}) == 128


def test_present_wins():
    assert _v_head_dim_of({"head_dim": 192, "v_head_dim": 128}) == 128


def test_neither_key_refuses_rather_than_assuming():
    with pytest.raises(KeyError, match="neither v_head_dim nor head_dim"):
        _v_head_dim_of({"block_m": 32})


def test_no_attention_at_all_refuses():
    with pytest.raises(ValueError):
        _v_head_dim_of(None)


# ------------------------------------- every reader gives the same answer

def test_the_eligibility_predicates_agree_on_an_absent_key():
    """The two predicates that gate the templates must read the same width
    whether the key is written out or left to mean symmetric."""
    without = _info()
    with_it = _info(v_head_dim=128)
    assert "v_head_dim" not in without
    assert _simd_fa_eligible(without) == _simd_fa_eligible(with_it)
    assert _fa_small_tile_eligible(without) == _fa_small_tile_eligible(with_it)


def test_an_absent_key_is_not_read_as_asymmetric():
    """The dissenting site's test, stated directly: absent must not make a
    symmetric kernel look asymmetric."""
    info = _info()
    assert _v_head_dim_of(info) == info["head_dim"], (
        "an absent v_head_dim read as anything other than head_dim sends a "
        "symmetric kernel down the asymmetric route")


def test_a_genuinely_asymmetric_kernel_still_reads_as_asymmetric():
    """The inertia: the fix must not flatten the case the key exists for."""
    info = _info(head_dim=192, v_head_dim=128, out_dtype="f16")
    assert _v_head_dim_of(info) != info["head_dim"]


def test_the_two_readings_transcribed_do_disagree():
    """The defect, written out rather than described.

    These two expressions are the code as it stood at the five sites. They
    are transcribed here because a test that imports the fix cannot fail on
    the tree without it — an ImportError demonstrates nothing. What it CAN
    do is show that the two readings answer differently on the same dict,
    and that the accessor now picks one of them, once.
    """
    info = {"head_dim": 128}

    four_sites = info.get("v_head_dim", info["head_dim"])   # "symmetric"
    fifth_site = info.get("v_head_dim")                     # None

    assert four_sites == info["head_dim"]
    assert fifth_site != info["head_dim"], (
        "None != 128 — this is how an absent key was read as asymmetric")

    assert _v_head_dim_of(info) == four_sites
