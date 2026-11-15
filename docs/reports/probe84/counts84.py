"""Memory traffic and arithmetic per cell, counted by reading the code (prompt 84).

The counting rule, applied to every kernel below:

- **Bytes** are those of an ideal fused kernel: every distinct cell or face
  array the kernel reads is read once and every array it produces for a
  later kernel is written once, 8 bytes per float64 element and 1 per bool.
  Temporaries NumPy forms inside a kernel are not counted (a port fuses
  them), nor are one-dimensional geometry arrays or scalars. That holds where
  the code broadcasts or multiplies a cell size, face distance, area or
  weight out to a full array (``np.broadcast_to``, ``* np.ones``): a port
  indexes it by row or column, so it is not a field. A real [ny, nx] array
  that geometry fills (the cell volumes, ``np.outer``) is one. A face array
  ([ny, nx+1] or [ny+1, nx]) counts as one element per cell. A kernel that
  must finish before the next can start (a sweep, a reduction the stop reads)
  is counted per execution, so an iterative stage's bytes are its sweep's
  bytes times the sweeps.
- **Operations** are floating-point additions, subtractions, multiplications,
  divisions, comparisons of floats, ``abs``, ``max``/``min``, ``sqrt``,
  ``log`` and a power, one each per element; ``x**3`` is two multiplications.
  A finiteness test (``isfinite``) counts as one comparison, as it does in
  every kernel that runs one. Selections (``np.where``), mask logic, copies
  and integer indexing are not operations. A reduction (``sum``, ``max``,
  ``vdot``) is one operation per element per term (``vdot`` two).

Counts are per cell per execution of the kernel. ``repeat`` names how many
times a stage's call executes it: a number, or a key ``tables84.py`` reads
from the measurement records (the conjugate gradient iterations of each
correction, the Jacobi sweeps of each class step). The derivation of each
count is in its ``how``: the arithmetic of the lines named, in the order the
code does it.

``scalars`` is the number of floating-point values a stage's call hands back
to Python that Python reads before the next kernel can start (a stop test, a
budget sum): what a port of only the inner loops would still read across the
device boundary each call.
"""

from dataclasses import dataclass

# Ada's FP64 rate (NVIDIA Ada GPU architecture whitepaper, p. 7: "The FP64
# TFLOP rate is 1/64th the TFLOP rate of FP32 operations").
FP64_FRACTION = 1.0 / 64.0
# The driver's attributes (cuDeviceGetAttribute, recorded in tables84's
# machine table): 36 SMs, a 128-bit bus at a memory clock of 8001 MHz, 32 MiB
# of L2. CUDA's peak bandwidth is 2 x memory clock x bus width / 8 (GDDR6
# transfers twice per clock), 256.0 GB/s, NVIDIA's own figure for this part.
SMS = 36
FP32_PER_SM_PER_CLOCK = 2 * 128  # 128 FP32 cores per SM, a fused multiply-add each
MEMORY_CLOCK_HZ = 8001e6
BUS_BITS = 128
L2_BYTES = 32 * 1024 * 1024
PEAK_BANDWIDTH = 2 * MEMORY_CLOCK_HZ * BUS_BITS / 8
# NVIDIA's boost clock range for the RTX 4070 Laptop GPU, 1230-2175 MHz
# (nvidia.com/en-us/geforce/laptops/40-series/, specs table); the laptop's
# power limit sets where in it the part runs.
BOOST_HZ = (1230e6, 2175e6)
# NVIDIA publishes no L2 bandwidth for this part. The one Ada measurement
# found is the RTX 4090's, about 5 TB/s with all 128 SMs active (Chips and
# Cheese, "Microbenchmarking Nvidia's RTX 4090", 2022-11-02, whose text adds
# that the figure exceeds the theoretical one at high occupancy because
# duplicate reads are combined). Scaled by SM count that is about 1.4 TB/s
# for 36 SMs, across dies whose L2 slices differ. These three are assumed,
# not measured, and bracket it; the report states every L2 verdict at each.
L2_BANDWIDTHS_ASSUMED = (1.0e12, 2.0e12, 4.0e12)
# The multigrid's coarsest level on all three grids: 2x2 aggregation with
# ceiling halves stops at COARSEST_CELLS = 64 (src/multigrid.py), so 80x30,
# 160x60 and 320x120 all end at 10x4 = 40 cells.
COARSEST_CELLS = 40


def peak_fp32(clock_hz: float) -> float:
    """Peak FP32 operations per second at a graphics clock (an FMA counted as two)."""
    return SMS * FP32_PER_SM_PER_CLOCK * clock_hz


def peak_fp64(clock_hz: float) -> float:
    """Peak FP64 operations per second at a graphics clock."""
    return peak_fp32(clock_hz) * FP64_FRACTION


@dataclass(frozen=True)
class Kernel:
    """One kernel of a stage: operations and bytes per cell per execution."""

    stage: str
    name: str
    ops: float
    bytes: float
    repeat: float | str
    how: str


# The k-epsilon boundary's full-field wall sides (TurbulenceBoundary._sides):
# read from the built object by tables84.py, since the count depends on the
# room's walls; each side evaluates the wall function over the whole field.
SIDES = "turbulence_sides"

KERNELS = [
    # ---- An outer iteration of the flow solve -----------------------------
    Kernel(
        "flow.momentum",
        "effective viscosity",
        4,
        17,
        1,
        "solver_staggered._effective_viscosity: isfinite, < 0 on live cells, "
        "mu + rho nu_t (2); reads nu_t and the live mask, writes mu_eff",
    ),
    Kernel(
        "flow.momentum",
        "viscosity check",
        2,
        9,
        1,
        "MomentumPredictor._check_mu_eff: isfinite and > 0 on live cells",
    ),
    Kernel(
        "flow.momentum",
        "assembly",
        108,
        92,
        2,
        "MomentumPredictor._assemble per component: mass fluxes f_s (3) and "
        "f_t (4); corner viscosity, weighted harmonic (8) and mean (3); "
        "conductances d_s, d_t and the obstacle faces (6); upwind a_e, a_w, "
        "a_n, a_s (10) and a_p (7); deferred QUICK correction, two face "
        "comparisons, four quadratics of five operations, the obstacle "
        "forms (6), two differences times flux and the four-term net (46); "
        "the stress source of a varying viscosity (20); its sum (1). Reads "
        "u, v, mu_eff, the wall viscosity and four masks; writes seven "
        "coefficient arrays",
    ),
    Kernel(
        "flow.momentum",
        "pressure source",
        3,
        16,
        2,
        "MomentumPredictor._pressure_source: -(p_+ - p_-) dt_cell",
    ),
    Kernel(
        "flow.momentum",
        "sweep setup",
        7,
        57,
        2,
        "_sweep_n: unknown mask (1), a_p / alpha (1), b from three sources "
        "and the relaxation term (5); reads a_p, b_boundary, b_deferred, "
        "b_pressure, phi; writes a_p_ur, b and the mask",
    ),
    Kernel(
        "flow.momentum",
        "Jacobi sweep",
        9,
        66,
        "momentum_sweeps_x2",
        "_sweep_n loop body: four products and four sums (8), one division; "
        "reads four neighbour coefficients, b, a_p_ur, the iterate and two "
        "masks; writes the next iterate",
    ),
    Kernel(
        "flow.pressure",
        "face d",
        4,
        32,
        2,
        "PressureCorrector._face_d (called by coefficients and again by "
        "correct): a_p > 0 and A / a_p per face, both components",
    ),
    Kernel(
        "flow.pressure",
        "coefficients",
        7,
        57,
        1,
        "PressureCorrector.coefficients: rho d A per face (2 x 2), a_p as "
        "the four-term sum (3); reads d_u, d_v, the SOLID mask; writes five "
        "coefficient arrays",
    ),
    Kernel(
        "flow.pressure",
        "right-hand side",
        7,
        42,
        1,
        "mass_imbalance of u*, v*: two differences, two products, a sum, "
        "times rho (6); f = -b with the inactive cells zeroed (1)",
    ),
    Kernel(
        "flow.pressure",
        "hierarchy",
        13,
        128,
        1,
        "Multigrid.__init__: dominance check (5) on the five coefficients; "
        "per level the smoother 2/3 / a_p (3) and coarsen's pair sums (3 per "
        "fine cell), summed over levels (x 4/3); writes each level's five "
        "arrays and smoother",
    ),
    Kernel(
        "flow.pressure",
        "coarsest pseudo-inverse",
        "pinv_ops",
        "pinv_bytes",
        1,
        "_dense_pseudo_inverse: eigh of the 40 x 40 matrix and V diag V^T, "
        "about 11 n^3 operations for n = 40 cells, on n^2 doubles written "
        "three times; per fine cell, so it falls with the grid",
    ),
    Kernel(
        "flow.pressure",
        "CG setup",
        2,
        24,
        1,
        "_conjugate_gradient_loop: ||f|| (2), r = f copy",
    ),
    Kernel(
        "flow.pressure",
        "CG iteration",
        21,
        136,
        "cg_iterations",
        "q = A p with p.q (11; reads five coefficients and p, writes q); "
        "x += alpha p, r -= alpha q with r.r (6; reads x, p, r, q, writes x, "
        "r); r.z and p = z + beta p (4; reads r, z, p, writes p)",
    ),
    Kernel(
        "flow.pressure",
        "V-cycle",
        33,
        229,
        "mg_cycles",
        "Multigrid._cycle, one sweep each side: x = w r (1), restrict of "
        "r - A x (10 + 0.75), prolong (1), post-smooth x += w (r - A x) "
        "(12): 24.75 per fine cell, x 4/3 over the levels; bytes likewise "
        "(172 x 4/3)",
    ),
    Kernel(
        "flow.pressure",
        "coarsest solve",
        "coarse_ops",
        "coarse_bytes",
        "mg_cycles",
        "the dense 40 x 40 product of _cycle's last level, 2 n^2 operations "
        "on n^2 doubles, per fine cell",
    ),
    Kernel(
        "flow.pressure",
        "true residual",
        12,
        64,
        "cg_checks",
        "the check of a stop on the true residual: f - A x and its norm",
    ),
    Kernel(
        "flow.pressure",
        "velocity correction",
        8,
        89,
        1,
        "u = u* - d (p'_+ - p'_-) both components (6; reads u*, v*, d_u, "
        "d_v, p'; writes u, v); p += alpha p' on active cells (2)",
    ),
    Kernel(
        "flow.turbulence",
        "checks",
        9,
        57,
        1,
        "KEpsilonModel._check_state, _check_faces, _check_conditions: "
        "finiteness and positivity of k, eps, nu_t, the faces and the "
        "wall arrays",
    ),
    Kernel(
        "flow.turbulence",
        "pseudo-time step",
        10,
        26,
        1,
        "pseudo_time_step: _rate's two abs, two max, two divisions and a sum "
        "(7), rate > 0 (1), cfl / rate (1) and the max of dt over the moving "
        "cells (1)",
    ),
    Kernel(
        "flow.turbulence",
        "source terms",
        31,
        83,
        1,
        "_terms with _strain_squared: dudx, dvdy (4), two corner "
        "differences (8), shear and its centre value (5), S^2 (5); "
        "production, eps/k, growth and decay of eps (9)",
    ),
    Kernel(
        "flow.turbulence",
        "advecting fluxes",
        2,
        34,
        1,
        "step: flux = velocity x area on both components, once for k and eps "
        "(2); reads the faces and two masks, writes the two flux arrays. The "
        "areas are broadcast views of cell sizes, not counted",
    ),
    Kernel(
        "flow.turbulence",
        "advection",
        66,
        74,
        2,
        "_advect for k and for eps: advective_flux on both axes (2 x 30, as "
        "the transport's), the four-term divergence (3) and q - dt div / V "
        "(3); reads the two flux arrays, q twice, the solid stencil, V and "
        "dt_cell, writes adv_u, adv_v and the result",
    ),
    Kernel(
        "flow.turbulence",
        "explicit growth",
        4,
        50,
        1,
        "k* + dt growth_k, eps* + dt growth_eps, eps held where given",
    ),
    Kernel(
        "flow.turbulence",
        "conductances",
        16,
        27,
        2,
        "_conductances for sigma_k and sigma_e: nu + nu_t / sigma (2), the "
        "weighted harmonic face value on both axes (10), times A / d (4); "
        "reads nu_t, the solid mask and the two interior-face masks, writes "
        "g_u, g_v. The areas, face distances and weights are one-dimensional "
        "geometry, not counted",
    ),
    Kernel(
        "flow.turbulence",
        "implicit setup",
        11,
        64,
        2,
        "implicit_step for k and for eps: V decay (1), diagonal sum (4), "
        "any (1), V / dt, a_P, b (3), max |b| (2)",
    ),
    Kernel(
        "flow.turbulence",
        "CG iteration",
        22,
        137,
        "keps_iterations",
        "scalar_scheme._conjugate_gradient body: product (9), p.q, "
        "c += alpha p, r -= alpha q, max |r| (8), z = r / a_P, r.z, the "
        "direction (5)",
    ),
    Kernel(
        "flow.turbulence",
        "true residual",
        12,
        57,
        "keps_checks",
        "true_residual at the start and at each stop of the two solves",
    ),
    Kernel(
        "flow.turbulence",
        "nu_t and relaxation",
        8,
        72,
        1,
        "positivity of k and eps (2), c_mu k^2 / eps (3), (1 - a) nu + a "
        "nu_new (3); the face copy FaceVelocities.copy_of (32 bytes)",
    ),
    Kernel(
        "flow.boundary",
        "stencil viscosity",
        22,
        27,
        1,
        "MomentumPredictor.stencil_viscosity: _corner_viscosity of both "
        "components (2 x 11); reads mu_eff and masks, writes two corner "
        "arrays",
    ),
    Kernel(
        "flow.boundary",
        "wall viscosity",
        0,
        32,
        1,
        "TurbulenceBoundary.wall_viscosity: copies the two corner arrays; "
        "the wall function itself runs on wall faces only",
    ),
    Kernel(
        "flow.boundary",
        "tangential means",
        4,
        18,
        1,
        "conditions: the two along-face means 0.5 (a + b)",
    ),
    Kernel(
        "flow.boundary",
        "wall side",
        20,
        9,
        SIDES,
        "conditions, per wall side over the whole field: _wall_cell_values "
        "(sqrt, products, max, log, the two quotients: 18) and the two "
        "masked sums (2); reads k and the side's mask. The wall distance is "
        "a row or column of cell sizes that the code multiplies out to "
        "[ny, nx] (``* ones``), one-dimensional geometry, not counted",
    ),
    Kernel(
        "flow.boundary",
        "held values",
        4,
        52,
        1,
        "conditions: the two _inflow maxima over live cells (2), eps_wall "
        "and production over the wall count (2)",
    ),
    Kernel(
        "flow.stopping",
        "velocity step",
        10,
        49,
        1,
        "to_cell_centers (4) and max |u_c - u_prev|, max |v_c - v_prev| (6) over FLUID",
    ),
    Kernel(
        "flow.stopping",
        "viscosity step and scale",
        6,
        26,
        1,
        "max |nu_t - nu_t_old| over live cells (3) and viscosity_scale, "
        "the FLUID median of nu_t (a partition, about 3 comparisons per "
        "element)",
    ),
    Kernel(
        "flow.stopping",
        "imbalance summary",
        10,
        25,
        "imbalance_summaries",
        "_imbalance_summary when the rule asks for it: mass_imbalance (6), "
        "abs, max and two sums (4)",
    ),
    # ---- A class step of the march ----------------------------------------
    Kernel(
        "class.remainder",
        "input checks",
        16,
        52,
        1,
        "solve_timestep's checks: the copy and isfinite of C_k (1); "
        "stable_dt with its own _advecting_velocities (13: one settling "
        "subtraction, the finiteness tests of u and v (2), four abs, two "
        "max, two divisions, a sum, the max over live cells); "
        "_check_eddy_viscosity (2)",
    ),
    Kernel(
        "class.advection",
        "advective fluxes",
        65,
        53,
        1,
        "_advecting_velocities (3: the settling subtraction and the "
        "finiteness tests of u and v), flux = velocity x area (2); "
        "advective_flux on each axis (2 x 30): the face direction (1), "
        "quick_face_values' two quadratics (10), limited_face_values' UMIST "
        "clamp (18: d_down, its test, r, psi_quick, 2r, (1 + 3r) / 4, the "
        "four-way minimum, the maximum, c_c + psi d / 2), flux x face (1). "
        "Without the finiteness tests the count is 63. Reads the faces (16), "
        "three masks (3), C once per axis (16) and the solid stencil (2); "
        "writes adv_u, adv_v (16). The areas are broadcast views of the cell "
        "sizes (solver_transport._area_u, _area_v), not counted",
    ),
    Kernel(
        "class.divergence",
        "divergence update",
        6,
        41,
        1,
        "the four-term divergence (3) and c - dt div / V (3)",
    ),
    Kernel(
        "class.source",
        "source",
        4,
        32,
        "source_steps",
        "c* += dt rate (2) and sum(rate V) dt (2); only on steps with the source on",
    ),
    Kernel(
        "class.implicit",
        "conductances",
        22,
        40,
        1,
        "_face_eddy_viscosity's weighted harmonic on both axes (2 x 8) and "
        "(D + nu_f / Sc_t) G_f (2 x 3); reads nu_t and the base "
        "conductances, writes g_u, g_v",
    ),
    Kernel(
        "class.implicit",
        "solve setup",
        10,
        56,
        1,
        "implicit_step: diagonal sum (4), any (1), V / dt, a_P, b (3), max |b| (2)",
    ),
    Kernel(
        "class.implicit",
        "Jacobi pass",
        14,
        49,
        "jacobi_passes",
        "_jacobi body, one residual per pass and the update on all but the "
        "last: neighbour sum (7), b + sum - a_P c (3), max |r| (2), "
        "c + r / a_P (2); the leftover's sum when the max-norm holds is "
        "fused here. Reads c, g_u, g_v, b, a_P and the SOLID mask, writes c",
    ),
    Kernel(
        "class.booking",
        "deposition and content",
        10,
        41,
        1,
        "_book_deposition: v_d dt (C_- + C_+) on both axes (6) and the "
        "obstacle sums (2); MassBudget.in_domain: C V and its sum over "
        "non-SOLID cells (2). The boundary sums run on edges only",
    ),
    # ---- Once per march step ----------------------------------------------
    Kernel(
        "step.arrival",
        "arrival maps",
        25,
        121,
        "started_events",
        "ArrivalTracker.update once an event has started: the five channel "
        "sums (15), the threshold comparisons (5), isnan of the map (5); "
        "reads the five fields, the live mask and the five maps, writes "
        "the maps",
    ),
]

# Scalars handed to Python per call that it reads before going on (see the
# module docstring), counted from the code.
SCALARS = {
    "flow.pressure": "3 per CG iteration (p.q, ||r||, r.z) and 2 per check",
    "flow.turbulence": "4 per CG iteration of each solve (p.q, max |r|, r.z, and "
    "the stop) and the positivity tests",
    "flow.stopping": "4 per outer iteration (du, dv, the viscosity step and "
    "scale) and 3 per imbalance summary",
    "class.remainder": "3 per class step (isfinite, the stable step, the nu_t check)",
    "class.implicit": "1 per Jacobi pass (max |r|) and the leftover",
    "class.booking": "about 22 per class step: eight inflow and outflow edge "
    "sums, the source, fourteen deposition sums, the content",
    "step.monitor": "20 sensor readings per step (4 sensors x 5 classes)",
}

# The stages each port carries (prompt 84's closing question). The plan's two
# kernels (docs/PROJECT_PLAN.md, Phase 6 deliverables): the pressure CG with
# its multigrid cycles, and the transport's advection-diffusion stencil. A
# whole-step port carries every stage of the step.
LOOP_PORT = {
    "flow.pressure": "the CG loop with its V-cycles (correct/cg_loop)",
    "class.advection": "advective_flux and its limiter",
    "class.divergence": "the divergence update",
    "class.implicit": "the implicit solve (implicit_step)",
}


# The same advection counted with other kernel boundaries (the module
# docstring's rule counts a kernel per array pass as the code makes them): an
# intensity near the FP64 balance moves with where a port fuses.
VARIANTS = [
    Kernel(
        "class.advection",
        "both axes in one kernel",
        65,
        45,
        1,
        "as 'advective fluxes', C read once instead of once per axis",
    ),
    Kernel(
        "class.advection",
        "fused with the divergence update",
        71,
        45,
        1,
        "advection and divergence in one kernel: reads the faces, three "
        "masks, C, the solid stencil and V; writes c* only (adv_u, adv_v "
        "stay on chip)",
    ),
]
