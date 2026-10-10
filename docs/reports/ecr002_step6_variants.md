# ECR-002 Step 6: k and eps by Upwind, and the Variant Check

**Date:** 2026-10-10
**Tree:** branch `feature/ecr002-turbulence-advection` from main at 87283de. Commit A adds the
turbulence key `advection_scheme` (`upwind` or `umist`, absent meaning `upwind`) and advects k and
eps with it; nothing else under `src/` changes. Every flow run goes through the committed
`StaggeredSolver` with the key absent, so the default holds, and every particle march through the
committed `TransportSolver` with the converged flow handed to it.
**Instruments:** `docs/reports/probe48/` (committed after this section, before any run):
`coupled48.py` (prompt 47's runner, its `control` arm, nothing substituted), `transport48.py`
(prompt 47's march), `tables48.py` (the tables, prompt 47's where they apply), `figures48.py`
(prompt 47's figures), `bounded48.py` and the launcher `run48.sh`. Commit A's checks are
`check48.py` (the laminar hashes) and `plant48.py` (the planted defects). Raw output under
`results/builder48/` (untracked).
**Order:** sections 1 to 4 were committed before any run of item 4, this file alone. Section 5
onward was written after the runs.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in the tables `tables48.py`
generates from the records; the sentences point at the tables.

## 1. The question

Alex's decisions of 2026-10-10 (ECR-002 section 8, step 6's note; ADR-012 C's note): k and eps
are advected by plain upwind by default, the product grid is 320x120 with 160x60 its coarser
check, and RNG is rerun under upwind on that pair. Two questions follow. Does the coupled model,
with k and eps advected by upwind, converge the product room on the exact grid pair for both
variants? And where both converge, does the variant change where particles collect? The answers
settle step 8's configuration and tell step 7 which schemes to run.

## 2. Method (fixed before the runs)

**The room and the settings** are prompt 47's (`docs/reports/ecr002_step6_comparative.md`,
section 2.1): `configs/clean_room_default.yaml` regridded, the turbulence section of prompt 46
(`cfl_number` 0.25, `alpha_turbulence` 0.7, `max_iter` 500, `tol` 1e-10, the baseline inlet
turbulence) with no `advection_scheme` key, the solver keys of step 5's measurement
(`error_estimate`, `momentum_sweeps` 10, `pressure_rtol` 1e-4, `max_simple_iter` 10,000), from
rest, one BLAS thread, the cell of the largest change recorded every outer iteration. The
classification is step 5's rule as prompt 47 applied it.

**Item 4, parity.** The standard variant on 80x30, 160x60 and 320x120 through the key. Prompt 47's
`upwind` arm passed `upwind=True` to the same `advective_flux` from a subclass of the model; the
key does the same from the configuration. Each row is compared with prompt 47's record of the arm
(`results/builder47/standard_GRID_upwind`): the stop, the outer count (1,046, 2,003 and 4,472),
the face hash (`28f437c84e3d5057` on 80x30) and, beyond what the prompt asks, every saved field
(u, v, k, eps, nu_t, p) and the residual history, bit for bit. A row that does not reproduce stops
the build.

**Item 5, the variants.** Standard and RNG on 160x60 and 320x120 under the default. The standard
rows are item 4's. Each converged run gets the three-panel figure (streamlines coloured by speed,
nu_t / nu on a log scale, k, the room's boundaries marked) and a row of the converged table; a
bounded run gets bounded44's characterisation.

**Item 6, the variant check.** On each exact grid where both variants converge, prompt 47's three
sources (S1 beside the door wall, S2 above the litho tool, S3 at the hood bench, 0.1 m squares) and
two classes (0.5 and 5 micrometres), marched as prompt 47 marched them (its section 2.5: transport
Courant number 0.4, the converged faces and eddy viscosity frozen, from a zero field, the steady
rule, 600 s cap). For each source and class: the largest deposition location under each variant,
the hotspots in common among the five largest 0.2 m segments, and the order of the sensors above
the floor (1e-6 of the source-cell concentration). The same for RNG between the two grids. The
standard variant's marches are rerun on this build's rooms, so the standard rows of every
comparison come from this build's records; they are compared with prompt 47's marches on the same
rooms. Counts and orders, not percentages.

## 3. Predictions

### 3.1 Orchestrator (2026-10-10; committed before item 4's runs)

- (a) The key reproduces probe 47's three upwind rows bit for bit.
- (b) RNG converges on both exact grids under upwind. Its 19-iteration cycle was the limiter's.
- (c) Between variants on each grid, for every source and class: the largest deposition location
  is the same segment, and at least four of the five hotspots are shared.
- (d) RNG agrees between grids as standard did: same largest location, at least four of five
  shared.

What each outcome means:

- (b) fails: RNG's cycle is not the limiter's, and the product runs the standard variant alone
  until VAL-020 decides between them.
- (c) fails: the variant decides the answer, and VAL-020 (the impinging jet) becomes the deciding
  validation before step 8.

### 3.2 Builder (2026-10-10, before item 4's runs; scored apart)

- (a) Holds, every saved field and the residual history included: the key and the arm make the
  same calls with the same arguments.
- (b) RNG converges on 160x60 and stays bounded on 320x120. Its cycle in prompt 47 had a velocity
  step about a twelfth of the scale and a per-cell imbalance tens of times its tolerance, larger
  than the standard variant's limiter cycle by orders, and on 200x75 (prompt 46) it was a real
  oscillation of about 4 cm/s beside the litho tool's west face; I read that as a flow-level
  oscillation that RNG's lower eddy viscosity in strained regions permits, which the face value
  of k and eps damps on the coarser grid and not on the finer one.
- (c) On every grid where both converge, the largest location is the same segment for every source
  and class, and at least four of five hotspots are shared for S1 and S2. For S3, whose last three
  hotspots are hood-bench segments at rates five orders below its first, I expect four of five
  for one class at least and do not predict the other.
- (d) Not scorable if (b)'s second half holds.

## 4. Stop conditions (from the prompt)

- Item 4's parity fails.
- A laminar hash moves (VAL-001, its stretched twin, VAL-002 against the base), or a transport
  gate test changes.
- A positivity error, or a capped pressure correction, in any run.
