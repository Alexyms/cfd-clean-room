# Phase 6 Step 2a: the Grid and the Explicit Flux (prompt 87)

**Date:** 2026-11-16
**Prompt:** 87 (`docs/prompts/cc-prompt-87-scalar-flux.md`)
**Branch:** `phase6/scalar-flux`
**Status:** predictions committed; nothing built or run yet.

This is the first half of Phase 6 step 2 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-017 C, D, F and G, decision 4). It builds the core's `grid` and `layout` as far as the
explicit face flux needs them, and ports the flux itself from `src/scalar_scheme.py` and
`src/momentum.py`: the QUICK face values with their Lagrange weights, the UMIST clamp, and the
advective flux per direction with its boundary nodes, its far-solid rule, `upwind` and the
vents. The implicit solve is prompt 88's. Nothing under `src/` or `configs/` changes.

## The predictions

Set by the orchestrator in prompt 87 on 2026-11-16, and committed alone before any run.

- (a) Every ported function equals the reference bit for bit on every planted and captured input,
  both directions, every option.
- (b) On the captured 320x120 inputs the core's `advective_flux` is at least ten times faster per
  call than the reference's, call overhead included.
- (c) The weights computed once at setup equal the reference's per-call weights bit for bit on
  all four meshes.

## How they are scored (the builder, before the runs)

- **(a).** Bit for bit means the 64-bit patterns are equal, so 0.0 and -0.0 differ and two NaNs
  are compared by pattern. The ported functions are `quick_face_values` (with the weights
  handed in), `limited_face_values` and `advective_flux`; each is called through `_cfdcore` and
  its reference on the same inputs, the reference as its callers call it (the y direction on
  the transposed arrays, `solver_transport.py` and `turbulence.py`), the core on the fields in
  their own [ny, nx] layout. The planted inputs are the parity tests' (`tests/test_core_parity.py`,
  section 3 of this report): four meshes (80x30, 160x60 and 320x120 from
  `configs/clean_room_default.yaml`, and a uniform mesh), both directions, `upwind` true and
  false, vents present (from `configs/clean_room_vents.yaml`) and absent. The captured inputs
  are every call the reference made in the probe's runs (section 4). The prediction holds when
  no element of any output differs in any of them; one differing element is a failure,
  reported with the function, mesh, direction and option.
- **(b).** Per call means the median wall time of repeated calls on one captured 320x120 input
  per direction, one thread, the core called through the binding (its call overhead included)
  and the reference as its callers call it (the transposes of the y call included, the vents'
  argument as the transport solver passes it). The prediction holds when the reference's median
  is at least ten times the core's in both directions. The empty call's overhead is reported
  beside it and does not change the score.
- **(c).** The reference's per-call weights are the six arrays `quick_face_values` computes
  from `nodes`, `left` and `faces` (`_lagrange_weights` for the positive and the negative
  direction), formed here with the same functions on each mesh's `mesh_axes` output, both
  directions. The core's are the ones its setup stores, read back through the binding. The
  prediction holds when all six arrays are equal bit for bit on all four meshes, both directions.

## 1. The design of `grid` and `layout`

The reference advects along the last axis: `mesh_axes` describes each direction with its nodes,
interior faces, left-node indices and a padded SOLID mask, and the y call transposes the field,
the flux, the inflow and the vent mask so that y is last (`solver_transport.py` and
`turbulence.py`, the `.T` calls). The core does not transpose. It keeps every field in the
reference's own row-major layout and walks it along a direction with that direction's stride.

**`layout<D>`** (`core/include/cfdcore/layout.hpp`) is the indexing, and nothing else.

- A cell field is one value per cell, row-major with direction 0 the fastest: [ny, nx] for
  D = 2. The face field normal to direction d is one value per face normal to d in the same
  order, with n_d + 1 entries along d: [ny, nx + 1] for d = 0 and [ny + 1, nx] for d = 1, the
  shapes the transport solver's `flux_u`, `flux_v`, `inflow_u` and `inflow_v` already have.
- `cell_stride(d)` is the offset from a cell to its high neighbour along d (1 and nx for
  D = 2); `face_stride(d)` the offset between adjacent faces normal to d along d, which is the
  same number, since the directions below d have the same extent in both fields.
- `low_face(d, cell)` takes a cell to its low face in the face field normal to d,
  `cell + (cell / (s_d n_d)) s_d`; the high face is one face stride beyond.
- `lines(d)` lists the lines of cells along d, each as its first cell and its low boundary
  face, built once at construction. For d = 1 they run i = 0 to nx - 1, which is the row order
  of the reference's transposed field, so line m of the core is row m of the reference's y call.
- `cell_field()` and `face_field(d)` return zeroed fields of the right length, for allocating
  once at set-up.

**`grid<D>`** (`core/include/cfdcore/grid.hpp`) is the geometry the flux reads, copied once
from the arrays `Mesh` holds: per direction the n_d + 2 nodes (the low boundary face, the cell
centres, the high boundary face, concatenated as `mesh_axes` concatenates them) and the n_d - 1
interior face coordinates; the cell types as given; and a SOLID mask made from them with the
SOLID code handed in (`mesh.SOLID`), so the core keeps no copy of the mesh's codes. `check`
returns the reason an input cannot make a grid (a direction with no cell, faces not one more
than centres, cell types not one per cell), which the binding raises as `TypeError` (ADR-017 B:
a wrong shape is refused at the call). Both are templates on `D`, instantiated at 1 and 2
(`core/src/grid.cpp`, `core/src/layout.cpp`); the binding builds `Grid2` from `x`, `xc`, `y`,
`yc`, `cell_type` and the SOLID code.

**Considered and not built**, because the flux does not read them:

- cell widths, centre distances, face areas and cell volumes (ADR-017 C's list for `grid`): the
  flux takes the volume flux per face, already times the area, as the reference does; the
  divergence update, which divides by the volume, is the transport step's;
- storage owned by `layout` for the resident fields (ADR-017 C: "the staggered field storage"):
  the flux's fields come from Python per call, so the resident storage waits for the run
  object, which owns it for a run;
- the strides of a face field along the directions other than its own, and a general
  neighbour lookup with boundary handling: the flux walks lines, so a cell's neighbour along the
  line and its two faces are all it needs;
- the boundary cell type (`BOUNDARY`) and the obstacle walls' faces: the flux reads SOLID only,
  and the boundary interpretation stays in Python (ADR-017 E);
- `std::mdspan` for the fields (not in GCC 13's library, ADR-017 A) and a binding that takes a
  `Mesh` object (the binding takes arrays, so the core knows nothing of the Python class);
- an instantiation at D = 3 (decision 4: no 3D feature built early). The templates take any
  `D >= 1`; nothing above 2 is compiled.
