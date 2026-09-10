"""A program_id read inside a region is still a program_id the kernel reads.

`IRGraph.ops` documents its own scope — "Top-level ops in topological order".
The walker collects region bodies separately and hangs them off their parent
as `region_ops` / `else_ops`. Five scans asked "which program-id axes does
this kernel use?" by iterating `graph.ops`, so a `tl.program_id(1)` written
inside an `scf.if` or `scf.for` was invisible to all of them.

What that costs, per scan:

  * the grid-convention scan: `axes - {0}` is empty, the kernel is declared
    to use a FLAT grid, `pid3` is emitted as a scalar and `pid_n` is derived
    from the flat id — every program computes a first-column tile and the
    rest of the output is never written. That is the exact defect the
    surrounding fix removes, returning through nesting.
  * `_refuse_if_pid_tiles_baked_output`: the refusal it exists to raise is
    skipped.
  * the two `has_pid` scans: one emits a kernel signature with no
    threadgroup position at all; the other skips a refusal about constexpr
    dims.
  * `_detect_*` constexpr-stride: answers "no constexpr stride" and the
    caller uses a template it should have declined.

These tests pin the scan itself, not one kernel, because the same blind spot
had five consumers.
"""
import pytest

from triton_msl.codegen.mlir_walker import IRGraph, SSAValue
from triton_msl.codegen.generic_lowerer import GenericLowerer


def _pid(sid, axis):
    return SSAValue(id=sid, name=f"v{sid}", op="tt.get_program_id",
                    operand_ids=[], attrs={"axis": axis}, type_str="i32",
                    elem_type="i32", is_tensor=False)


def _region(sid, op, body=None, els=None):
    return SSAValue(id=sid, name=f"v{sid}", op=op, operand_ids=[], attrs={},
                    type_str="", elem_type="", is_tensor=False,
                    region_ops=body, else_ops=els)


def _lowerer(ops):
    g = IRGraph(func_name="t", args=[], ops=ops, mod_text="")
    lo = GenericLowerer.__new__(GenericLowerer)
    lo.graph = g
    return lo


# ---------------------------------------------------------------- the walk

def test_the_walk_reaches_a_region_body():
    from triton_msl.codegen.mlir_walker import iter_ops_recursive
    inner = _pid(2, 1)
    ops = [_pid(1, 0), _region(3, "scf.for", body=[inner])]
    assert inner in list(iter_ops_recursive(ops))


def test_the_walk_reaches_an_else_body():
    from triton_msl.codegen.mlir_walker import iter_ops_recursive
    inner = _pid(2, 1)
    ops = [_region(3, "scf.if", body=[], els=[inner])]
    assert inner in list(iter_ops_recursive(ops))


def test_the_walk_reaches_a_region_inside_a_region():
    from triton_msl.codegen.mlir_walker import iter_ops_recursive
    inner = _pid(4, 1)
    ops = [_region(3, "scf.for", body=[_region(5, "scf.if", body=[inner])])]
    assert inner in list(iter_ops_recursive(ops))


def test_the_walk_tolerates_an_op_with_no_regions():
    from triton_msl.codegen.mlir_walker import iter_ops_recursive
    ops = [_pid(1, 0)]
    assert [o.id for o in iter_ops_recursive(ops)] == [1]


# ------------------------------------------- the grid convention, the point

def test_a_nested_axis_1_makes_the_grid_multi_axis_not_flat():
    """The scan that decides flat-vs-multi-axis must see into the region.

    Fails on the unfixed tree: `axes - {0}` is empty, the method reports a
    flat grid, and `pid_n` is then derived from an id that carries only the
    row tile.
    """
    ops = [_pid(1, 0), _region(3, "scf.if", body=[_pid(2, 1)])]
    lo = _lowerer(ops)
    lines = []
    flat = lo._emit_tile_ids_from_source_grid(lines, None)
    assert flat is False, "a kernel reading program_id(1) is not on a flat grid"
    assert lo._used_pid_axes == {0, 1}
    assert any("uint3 pid3" in l for l in lines), lines


def test_a_kernel_that_really_is_flat_stays_flat():
    """The inertia: nothing changes for a kernel whose only axis is 0, at
    top level or nested."""
    ops = [_pid(1, 0), _region(3, "scf.for", body=[_pid(2, 0)])]
    lo = _lowerer(ops)
    lines = []
    flat = lo._emit_tile_ids_from_source_grid(lines, None)
    assert flat is True
    assert lo._used_pid_axes == {0}
    assert any(l.strip().startswith("uint pid3") for l in lines), lines


# ------------------------------------------------- the refusal that was skipped

def test_the_baked_output_refusal_sees_a_nested_axis():
    """`_refuse_if_pid_tiles_baked_output` raises when the kernel tiles the
    output across programs while M/N are constexpr. A nested program_id made
    it stay silent."""
    from triton_msl.errors import MetalNonRecoverableError

    ops = [_region(3, "scf.if", body=[_pid(2, 1)])]
    lo = _lowerer(ops)
    with pytest.raises(MetalNonRecoverableError):
        lo._refuse_if_pid_tiles_baked_output(has_M=True, has_N=False,
                                             what="strided matmul")
