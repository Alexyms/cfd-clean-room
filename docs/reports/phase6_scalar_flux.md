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
