"""The staged fill must carry the load's mask BEFORE the wrap is moved inside.

Two defects live behind one refusal, and their order is not a preference.

A `tt.dot` whose operand is staged through threadgroup memory, inside an
`scf.for`, with a tile wider than the threadgroup, is REFUSED today: the
cooperative fill needs a loop outside the wrap, while the operand's address
depends on the k-loop's induction variable, so the staging would have to be
inside the k-loop and outside the wrap at once. The remedy is to move the wrap
INSIDE the k-loops, which the multipass-reduction machinery in this file
already knows how to do for the wide scan.

But the staged fill DROPS THE LOAD'S MASK. It emits

    shared[_sa] = ptr[<rebuilt offset>];

where the in-loop form emits `mask ? ptr[addr] : 0`. On a boundary tile that
reads past the end of the tensor, and the value reaches shared memory and then
the dot. The store side already refuses what it cannot reconstruct
(`_rebuild_staged_fill_mask`); the fill side does not consult the mask at all.

**So the refusal is, today, the only thing preventing a silent wrong.**
Repairing the nesting first would trade a loud refusal for a quiet wrong
answer — the one exchange this project does not make.

This test exists to make that order MANDATORY rather than remembered. It is
red in exactly one situation: the refusal is gone AND the fill still ignores
the mask. It is green while the refusal stands, and green once the fill is
masked, in either order of arrival — but never for the combination that ships
a silent wrong.
"""
import inspect
import re

import pytest

from triton_msl.codegen import generic_lowerer as GL


#: The refusal that currently stands in for the missing mask.
_REFUSAL = "per-element wrap loop is emitted OUTSIDE that loop"

#: Every emission of the cooperative staged fill: one shared-memory element per
#: `_sa` step, read from GLOBAL memory. `{src_var}` sites are excluded — they
#: stage a value already loaded per-thread and read nothing out of bounds.
_FILL = re.compile(r'raw_line\(\s*f?"[^"]*\{shared_name\}\[_sa\] = (?P<rhs>[^"]+)"')

#: The guard helper the fill sites must pass their value through.
#:
#: The first version of this detector read the EMISSION LINE and asked whether
#: it contained a `?`. The masking happens in a helper, so the line only ever
#: says `{val_expr};` — no `?`, no `0` — and every assertion below passed for
#: the wrong reason. A detector blind to one level of indirection is a vacuous
#: guard, which is the class this file exists to stop.
_GUARD_HELPER = "_guarded("


def _source():
    return inspect.getsource(GL)


def _fill_sites(src):
    return [m.group("rhs") for m in _FILL.finditer(src)]


def _global_fill_sites(src):
    """The fill sites that read global memory — the ones that can go OOB."""
    return [rhs for rhs in _fill_sites(src) if "{src_var}" not in rhs]


def _guard_body(src):
    """The body of the guard helper, or None when there is none."""
    marker = "def _guarded("
    if marker not in src:
        return None
    i = src.index(marker)
    return src[i:i + 800]


def _fill_is_masked(src, rhs):
    """A global fill is masked when its value passed through the guard helper.

    Either literally on the emission line, or via the helper applied to the
    expression it emits — the indirection the first version missed.
    """
    if "?" in rhs or "_fill_mask" in rhs or "_sa_mask" in rhs:
        return True
    if _GUARD_HELPER not in src:
        return False
    # the value written is built by the guard helper somewhere in this emitter
    return src.count(_GUARD_HELPER) >= len(_global_fill_sites(src))


def _guard_honours_other(src):
    """False when the guard helper writes a literal 0 for the rejected lanes.

    `tt.load(ptr, mask, other)` has THREE operands and `other` is the third:
    the value the mask's rejected lanes are declared to take. `_lower_load`
    honours it in the in-loop path. A staged fill that selects on the mask and
    hands back a hardcoded 0 gives the dot a different tile than the kernel
    asked for — silently, and only where `other` is not zero, which is exactly
    where nobody looks.
    """
    body = _guard_body(src)
    if body is None:
        return False
    return not re.search(r"\?\s*\{?[^:}]*\}?[^:]*:\s*0[^\w]", body)


def test_the_staged_fill_sites_are_found_at_all():
    """A detector that finds nothing would make every assertion below vacuous."""
    src = _source()
    assert _fill_sites(src), (
        "no cooperative staged-fill emission found — this test can no longer "
        "see what it guards, which is worse than failing")
    assert _global_fill_sites(src), "no global-reading fill site found"


def test_the_refusal_and_the_unmasked_fill_cannot_both_be_absent():
    """THE ORDER, made mandatory.

    Red only for: refusal removed AND fill still unmasked. That is the state
    in which a boundary tile reads past its tensor and the dot consumes it,
    with nothing said.
    """
    src = _source()
    refusal_stands = _REFUSAL in src
    unmasked = [rhs for rhs in _global_fill_sites(src) if not _fill_is_masked(src, rhs)]

    assert refusal_stands or not unmasked, (
        "the refusal that stood in for the missing mask has been removed while "
        f"{len(unmasked)} staged-fill site(s) still ignore the load's mask:\n  "
        + "\n  ".join(unmasked)
        + "\n\nThe wrap was moved inside the k-loop before the fill was masked, "
          "which trades a loud refusal for a silent out-of-bounds read. Mask "
          "the fill first."
    )


def test_a_masked_fill_does_not_hardcode_zero_for_the_rejected_lanes():
    """The fifth row of the matrix, and the one that can be committed WHILE
    fixing the fourth.

    `tt.load(ptr, mask, other)`: the rejected lanes take `other`, not zero.
    A fill that selects on the mask and writes a literal 0 is wrong wherever
    `other` is not zero — and a tile of zeros is indistinguishable from an
    accumulation that ignores its masked positions, so this cannot be caught
    by looking at a result.
    """
    # NO skip while the refusal stands. The emitter's correctness does not
    # depend on whether a refusal currently prevents reaching it, and a row
    # that sleeps until the refusal goes is a row that wakes up the day the
    # defect ships — which is the whole failure mode this file exists to stop.
    src = _source()
    assert _guard_honours_other(src), (
        "the staged fill selects on the mask but hands the rejected lanes a "
        "hardcoded 0 instead of the load's declared `other`. A tile of zeros "
        "is indistinguishable from an accumulation that ignores its masked "
        "positions, so no result will show this.\n"
        + (_guard_body(src) or "")[:400])


def test_the_store_side_still_refuses_what_it_cannot_reconstruct():
    """The half that is already right, pinned so it is not loosened to make the
    fill side's job easier."""
    src = inspect.getsource(GL.MSLGenerator._rebuild_staged_fill_mask) \
        if hasattr(GL, "MSLGenerator") else _source()
    assert "MetalNonRecoverableError" in src or "_refuse" in src, (
        "the cooperative store's mask rebuild no longer refuses an "
        "unreconstructable mask")


def test_the_refusal_names_the_remedy_not_just_the_limit():
    """A refusal that names only the ceiling teaches nothing; this one names
    the nesting, which is what makes the chantier tractable."""
    src = _source()
    if _REFUSAL not in src:
        pytest.skip("the refusal is gone; the fill-mask assertion covers it")
    assert "wrap moved inside the k-loops" in src


# ---------------------------------------------------------------------------
# The test proves it can fail
# ---------------------------------------------------------------------------
#
# A guard that cannot go red guards nothing, and this file exists to be red in
# situations nobody reaches by accident. They are constructed here, from source
# strings, and the assertions are checked against them directly.

_EMIT = 'raw_line(f"        {shared_name}[_sa] = %s;")'
_UNMASKED = _EMIT % "{base_ptr}[{new_offset}]"
_VIA_HELPER = (_EMIT % "{val_expr}") + '\n        val_expr = _guarded(x, g)\n'
_GUARD_ZERO = 'def _guarded(expr, guard):\n    return expr if guard is None else f"({guard} ? ({expr}) : 0)"\n'
_GUARD_OTHER = 'def _guarded(expr, guard, other):\n    return expr if guard is None else f"({guard} ? ({expr}) : {other})"\n'
_WITH_REFUSAL = "per-element wrap loop is emitted OUTSIDE that loop"


def _order_verdict(src):
    """`test_the_refusal_and_the_unmasked_fill_cannot_both_be_absent`, as a
    value rather than an assert."""
    return (_REFUSAL in src) or not [
        rhs for rhs in _global_fill_sites(src) if not _fill_is_masked(src, rhs)]


def test_the_dangerous_combination_is_the_only_red_one():
    assert _order_verdict(_WITH_REFUSAL + _UNMASKED) is True, (
        "refusal present, fill unmasked: today's state before the fix, safe")
    assert _order_verdict(_WITH_REFUSAL + _VIA_HELPER + _GUARD_ZERO) is True, (
        "refusal present, fill masked: safe")
    assert _order_verdict(_VIA_HELPER + _GUARD_ZERO) is True, (
        "refusal gone, fill masked: the goal for THIS row")
    assert _order_verdict(_UNMASKED) is False, (
        "refusal gone, fill unmasked: the silent OOB read — MUST be red")


def test_a_hardcoded_zero_is_seen_as_ignoring_other():
    """The fifth row proves it can fail too.

    The synthetic `other` is a VARIABLE, not zero: a source where `other`
    happens to be 0 would be green for the wrong reason, a tile of zeros being
    indistinguishable from an accumulation that ignores its masked positions.
    """
    assert _guard_honours_other(_GUARD_OTHER) is True, (
        "a guard that names the traced `other` must pass")
    assert _guard_honours_other(_GUARD_ZERO) is False, (
        "a hardcoded 0 must be seen as ignoring `other` — otherwise this row "
        "is green for the wrong reason")
    assert _guard_honours_other("no helper here") is False, (
        "no guard helper at all cannot count as honouring `other`")


def test_the_detector_follows_one_level_of_indirection():
    """The failure that made this rewrite necessary: the masking happens in a
    helper, so the emission line only ever says `{val_expr};` — no `?`, no
    `0` — and a detector reading that line passed for the wrong reason."""
    assert _global_fill_sites(_VIA_HELPER), "the emission is not recognised"
    assert "?" not in _global_fill_sites(_VIA_HELPER)[0], (
        "the premise: the emission line carries no selection of its own")
    assert _fill_is_masked(_VIA_HELPER + _GUARD_ZERO, _global_fill_sites(_VIA_HELPER)[0]), (
        "the detector does not follow the value into the guard helper")


def test_a_per_thread_fill_is_not_a_global_read():
    """`{src_var}` stages a value already loaded per thread; it cannot go out
    of bounds and must not be demanded a mask."""
    per_thread = _EMIT % "{src_var}"
    assert _fill_sites(per_thread), "the site is not seen at all"
    assert not _global_fill_sites(per_thread), (
        "a per-thread staging was counted as a global read")


# ── the refusal's condition must be about the VALUE, not its registration ──


def _other_refusal_condition(src):
    """The `if` that guards the per-element-`other` refusal, as text.

    Returns None when no refusal is present at all, which the caller must
    treat as a failure rather than as "nothing to check".
    """
    needle = "cooperative staged fill of a masked load whose `other` is "
    at = src.find(needle)
    if at < 0:
        return None
    # Walk BACKWARDS to the nearest `if`, rather than matching forward across
    # an unbounded span. A forward regex with `(?:[^\n]*\n)*?` in it matched
    # an `if not op_name:` two hundred lines earlier and reported that as the
    # refusal's condition -- reading where the text merely resembles what is
    # wanted instead of where it actually governs.
    for line in reversed(src[:at].splitlines()):
        stripped = line.strip()
        if stripped.startswith("if ") and stripped.endswith(":"):
            return stripped[3:-1]
    return None


_ENV_ARRAY_ONLY = (
    "\n            if load_other_id in self.env_array:\n"
    "                from triton_msl.errors import MetalNonRecoverableError\n"
    "                raise MetalNonRecoverableError(\n"
    '                    "cooperative staged fill of a masked load whose `other` is "\n'
)
_WITH_UNIFORMITY = (
    "\n            if load_other_id in self.env_array or load_other_id not in self._is_splat:\n"
    "                from triton_msl.errors import MetalNonRecoverableError\n"
    "                raise MetalNonRecoverableError(\n"
    '                    "cooperative staged fill of a masked load whose `other` is "\n'
)


def test_the_other_refusal_exists_at_all():
    src = _source()
    assert _other_refusal_condition(src) is not None, (
        "no refusal for a non-uniform `other` was found in the lowerer. Every "
        "assertion below would then pass over an absent guard.")


def test_the_refusal_does_not_hang_on_the_mept_registration_alone():
    """`env_array` is populated only when `mept_enabled`.

    `TRITON_MSL_MEPT=0` restores the legacy scalar path and leaves `env_array`
    empty by construction — while the cooperative staged fill is still reached
    there, which running any staged-dot kernel under TRITON_MSL_MEPT=0 shows. A
    refusal whose only condition is `in self.env_array` is therefore dead code
    on the MEPT=0 path, and dead exactly where its own docstring promises to
    protect. The condition must also ask something about the VALUE.
    """
    cond = _other_refusal_condition(_source())
    assert cond is not None
    assert "_is_splat" in cond, (
        f"the refusal's condition is `{cond}`, which reduces to a property of "
        f"the MEPT lowering. With TRITON_MSL_MEPT=0 it can never be true and "
        f"the staged fill would substitute this thread's value for elements it "
        f"never loaded — the silent wrong this file exists to stop.")


def test_the_detector_rejects_the_env_array_only_form():
    """Both directions, on the two forms that actually existed.

    Without this row the assertion above is satisfiable by a detector that
    answers yes to any text: the first version of this guard WAS the
    env-array-only form, and it must come back red.
    """
    assert _other_refusal_condition(_WITH_UNIFORMITY) is not None
    assert "_is_splat" in _other_refusal_condition(_WITH_UNIFORMITY)
    cond = _other_refusal_condition(_ENV_ARRAY_ONLY)
    assert cond is not None, "the detector must still FIND the older form"
    assert "_is_splat" not in cond, (
        "the env-array-only form must be seen as lacking a value test; a "
        "detector that cannot tell the two apart pins nothing")
