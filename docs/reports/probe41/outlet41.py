"""Builder probe, prompt 41 (ECR-002 step 3): the pressure outlets under five rules.

Usage:
    python outlet41.py drift ARM RTOL [--n-outer N] [--hood-gradient] [--tag TAG]
    python outlet41.py ladder ARM RUNG [--n-outer N]       RUNG is 895 or 8950
    python outlet41.py val001 ARM [--grid NX NY]           ARM is committed, B, C or F
    python outlet41.py control                             sweep predictor vs frozen34
    python outlet41.py compare NAME_1 NAME_2 [--grid NX NY] measurement 4

ARM is one of A, B, C, D, E, E0 (section 2 of the report), F (section 8), or
Aopen and D0 (section 5.3: A's copy with the hold-shut rule off; D with the
returns' tangential velocity held at zero). Nothing under src/ is edited or
patched: the arms override StaggeredSolver's outlet
extrapolation, the corrector is a subclass that records its right-hand
side, and the ten-sweep predictor is a subclass of MomentumPredictor. Every
run writes NAME.json and NAME.npz under results/builder41/.

The drift case is the room of docs/reports/pressure_solver_ecr003.md
section 8.3: configs/clean_room_default.yaml on 80x30 at a thousand times
air's viscosity, alpha_velocity 0.5, ten momentum sweeps, from rest, under
error_estimate with ADR-011 G's per-cell bound. The ladder is prompt 33b's:
40x15, alpha_velocity 0.5, one momentum sweep, velocity_step, to 3,000 outer
iterations or 100 m/s. VAL-001 is the harness preset as its file configures
it. The hood holds 0.5 m/s outward with zero tangential velocity (test 33b's
tangD) unless --hood-gradient keeps the pressure outlet's zero gradient.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.boundary_registry import BoundaryRegistry  # noqa: E402
from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import FLUID, SOLID, Mesh  # noqa: E402
from src.momentum import (  # noqa: E402
    MomentumCoefficients,
    MomentumPrediction,
    MomentumPredictor,
)
from src.pressure import PressureCorrection, PressureCorrector  # noqa: E402
from src.solver_staggered import StaggeredSolver  # noqa: E402
from src.staggered import FaceVelocities, edge_cell_inputs  # noqa: E402
from src.stopping import IterationState  # noqa: E402
from validation.cases import load_case, load_preset  # noqa: E402
from validation.metrics import poiseuille_l2_error  # noqa: E402

OUT = ROOT / "results" / "builder41"
DIVERGED_SPEED = 100.0
HOOD_SPEED = 0.5
LADDER_FACTORS = {"895": 100.0, "8950": 10.0}
RULES = {
    "A": "copy",
    "B": "stick",
    "C": "local",
    "D": "fixed",
    "E": "stick",
    "E0": "copy",
    "F": "scaled",
    "Aopen": "copy_open",
    "D0": "fixed",
}


class DivergedError(Exception):
    """Raised from the callback to end a run that has diverged."""


def segment_names(mesh: Mesh, cfg: SimConfig, edge: str) -> np.ndarray:
    """The covering segment's name for every cell along an edge, '' for none."""
    coords, solid = edge_cell_inputs(mesh, edge)
    cover = BoundaryRegistry(cfg).coverage_along(edge, coords, solid)
    return np.array([c.name or "" for c in cover])


def hold_tangential(boundary: StaggeredBoundary, edge: str) -> int:
    """Every non-Dirichlet tangential location along an edge becomes Dirichlet zero.

    Test 33b's tangD, for any edge. Returns the number of locations switched.
    On the product configuration the right edge's are the hood's (the only
    pressure outlet there) and the bottom edge's are the four returns'. The
    wrapper composes: a second call on another edge keeps the first.
    """
    original = boundary.tangential_conditions
    condition = original()[edge]
    switched = int((~condition.is_dirichlet).sum())

    def wrapped() -> dict:
        out = dict(original())
        c = out[edge]
        value = np.array(c.value)
        value[~c.is_dirichlet] = 0.0
        out[edge] = replace(c, is_dirichlet=np.ones_like(c.is_dirichlet), value=value)
        return out

    boundary.tangential_conditions = wrapped  # type: ignore[method-assign]
    return switched


def hold_hood_tangential(boundary: StaggeredBoundary) -> int:
    """The hood's tangential velocity held at zero: hold_tangential on the right edge."""
    return hold_tangential(boundary, "right")


# ---------------------------------------------------------------------------
# The predictor with N sweeps (frozen34.py's _sweep_n on the committed assembly)
# ---------------------------------------------------------------------------


class SweepPredictor(MomentumPredictor):
    """The committed predictor with N Jacobi sweeps per outer iteration.

    N = 1 is the committed predict call. N > 1 runs frozen34.py's _sweep_n:
    N sweeps of the under-relaxed equations on the coefficients and sources
    of the current field, the relaxation term held at the outer iterate.
    """

    def __init__(
        self, mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary, sweeps: int
    ) -> None:
        super().__init__(mesh, config, boundary)
        self._n_sweeps = sweeps

    def predict(
        self, u: np.ndarray, v: np.ndarray, p: np.ndarray
    ) -> MomentumPrediction:
        """u*, v* and the diagonals after N sweeps; the committed call at N = 1."""
        if self._n_sweeps == 1:
            return super().predict(u, v, p)
        self._check_shapes(u, v)
        c_u = self._assemble(u, v, self._for_u)
        u_star = self._sweep_n(u, c_u, self._pressure_source(p, self._for_u))
        c_vt = self._assemble(u=v.T, v=u.T, o=self._for_v)
        v_star = self._sweep_n(v.T, c_vt, self._pressure_source(p.T, self._for_v)).T
        return MomentumPrediction(
            u_star=np.ascontiguousarray(u_star),
            v_star=np.ascontiguousarray(v_star),
            a_p_u=c_u.a_p,
            a_p_v=np.ascontiguousarray(c_vt.a_p.T),
        )

    def _sweep_n(
        self, phi: np.ndarray, c: MomentumCoefficients, b_pressure: np.ndarray
    ) -> np.ndarray:
        """N Jacobi sweeps, frozen34.py's lines."""
        alpha = self._alpha
        unknown = c.a_p > 0.0
        a_p_ur = np.where(unknown, c.a_p / alpha, 1.0)
        b = (
            c.b_boundary
            + c.b_deferred
            + b_pressure
            + (1.0 - alpha) / alpha * c.a_p * phi
        )
        interior = np.zeros_like(unknown)
        interior[:, 1:-1] = True
        current = phi
        for _ in range(self._n_sweeps):
            padded = np.pad(current, ((1, 1), (1, 1)))
            numerator = (
                c.a_s_plus * padded[1:-1, 2:]
                + c.a_s_minus * padded[1:-1, :-2]
                + c.a_t_plus * padded[2:, 1:-1]
                + c.a_t_minus * padded[:-2, 1:-1]
                + b
            )
            nxt = phi.copy()
            nxt[interior] = np.where(unknown, numerator / a_p_ur, 0.0)[interior]
            current = nxt
        return current


# ---------------------------------------------------------------------------
# The corrector that records its right-hand side
# ---------------------------------------------------------------------------


@dataclass
class CorrectionRecord:
    """What one correction saw: ||b||, its share beside open outlet faces, the solve."""

    b_norm: float
    b_share_open: float
    iterations: int
    reached_cap: bool
    needs_pin: bool


class RecordingCorrector(PressureCorrector):
    """PressureCorrector unchanged, plus a record of b per call and the corrected faces."""

    def __init__(
        self, mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary
    ) -> None:
        super().__init__(mesh, config, boundary)
        self.records: list[CorrectionRecord] = []
        self.last_b: np.ndarray | None = None
        self.last_u: np.ndarray | None = None
        self.last_v: np.ndarray | None = None

    def open_outlet_cells(self) -> np.ndarray:
        """Cells beside an outlet face the current masks leave open, shape [ny, nx]."""
        cells = np.zeros(self._p_shape, dtype=bool)
        cells[:, 0] |= self._out_left
        cells[:, -1] |= self._out_right
        cells[0, :] |= self._out_bottom
        cells[-1, :] |= self._out_top
        return cells

    def correct(
        self, prediction: MomentumPrediction, p: np.ndarray
    ) -> PressureCorrection:
        """The committed correction, with b and the corrected faces kept."""
        b = self.mass_imbalance(prediction.u_star, prediction.v_star)
        total = float(np.sum(b * b))
        beside = float(np.sum((b * b)[self.open_outlet_cells()]))
        result = super().correct(prediction, p)
        self.records.append(
            CorrectionRecord(
                b_norm=math.sqrt(total),
                b_share_open=beside / total if total > 0.0 else 0.0,
                iterations=result.iterations,
                reached_cap=result.reached_cap,
                needs_pin=self.needs_pin,
            )
        )
        self.last_b, self.last_u, self.last_v = b, result.u, result.v
        return result


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """One outlet segment and the rule its faces follow.

    rule is "copy" (arm A), "stick" (B), "local" (C), "fixed" (a held
    outward normal speed, arms D, D0 and the hood), "scaled" (arm F, found
    on the way: the copy, scaled with every other scaled segment so the
    outflow equals the supply, then held for the correction) or "copy_open"
    (arm A-open: the copy with no face ever held shut); mask is the
    segment's cells along its edge.
    """

    name: str
    edge: str
    mask: np.ndarray
    rule: str
    speed: float = 0.0


def face_row(u: np.ndarray, v: np.ndarray, edge: str) -> np.ndarray:
    """The edge's normal faces as a writable view."""
    if edge == "bottom":
        return v[0, :]
    if edge == "right":
        return u[:, -1]
    raise ValueError(f"edge {edge!r} is not handled by this probe")


def interior_row(u: np.ndarray, v: np.ndarray, edge: str) -> np.ndarray:
    """The interior faces one cell in from the edge."""
    return v[1, :] if edge == "bottom" else u[:, -2]


def inward(values: np.ndarray, edge: str) -> np.ndarray:
    """True where a normal face value points into the room."""
    return values > 0.0 if edge == "bottom" else values < 0.0


def outward_value(speed: float, edge: str) -> float:
    """The signed face value of an outward normal speed."""
    return -speed if edge == "bottom" else speed


def closing_velocity(u: np.ndarray, v: np.ndarray, edge: str, mesh: Mesh) -> np.ndarray:
    """The edge face value that closes each edge cell given its other three faces."""
    if edge == "bottom":
        return v[1, :] + (u[0, 1:] - u[0, :-1]) * mesh.dy_cell[0] / mesh.dx_cell
    return u[:, -2] - (v[1:, -1] - v[:-1, -1]) * mesh.dx_cell[-1] / mesh.dy_cell


class ArmSolver(StaggeredSolver):
    """StaggeredSolver with the outlet faces written by the arm's rules each outer iteration.

    Before each prediction every segment writes its faces: a fixed segment
    its outward speed; a copy segment the interior neighbour; a stick segment
    the corrected value of the previous iteration (the interior copy on the
    first iteration and on a face held shut the iteration before); a local
    segment the value that closes its cell. On the pressure segments a face
    whose written value points into the room is held at zero and left out of
    the corrector's outlet masks for that iteration. When no outlet face is
    open the corrector is told to pin, the closed-domain path.
    """

    def __init__(
        self,
        mesh: Mesh,
        cfg: SimConfig,
        boundary: StaggeredBoundary,
        segments: list[Segment],
        sweeps: int,
    ) -> None:
        super().__init__(mesh, cfg, boundary)
        self._predictor = SweepPredictor(mesh, cfg, boundary, sweeps)
        self._corrector = RecordingCorrector(mesh, cfg, boundary)
        self.segments = segments
        self._base = {
            edge: self._outlets[edge].is_outlet.copy() for edge in ("bottom", "right")
        }
        for edge in ("left", "top"):
            if self._outlets[edge].is_outlet.any():
                raise ValueError(f"a {edge} outlet is not handled by this probe")
        for edge in ("bottom", "right"):
            covered = np.zeros_like(self._base[edge])
            for seg in segments:
                if seg.edge == edge:
                    covered |= seg.mask
            if not np.array_equal(covered, self._base[edge]):
                raise ValueError(f"the segments do not cover the {edge} outlet faces")
        fluid = np.argwhere(mesh.cell_type == FLUID)
        self._pin = (int(fluid[0, 0]), int(fluid[0, 1]))
        self.closed = {edge: np.zeros_like(self._base[edge]) for edge in self._base}
        self.open = {edge: self._base[edge].copy() for edge in self._base}
        self.iteration = 0
        self.pinned_iterations = 0
        self.scale_history: list[float] = []
        self.fallback_iterations = 0
        self._supply = boundary.get_total_inlet_flux()

    def solve_steady(
        self, on_iteration: Callable[[IterationState], None] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The committed loop with the arm's counters reset."""
        self.iteration = 0
        self.pinned_iterations = 0
        self.scale_history = []
        self.fallback_iterations = 0
        self.closed = {edge: np.zeros_like(self._base[edge]) for edge in self._base}
        return super().solve_steady(on_iteration=on_iteration)

    def _widths(self, edge: str) -> np.ndarray:
        """Face widths along an edge."""
        return self._mesh.dx_cell if edge == "bottom" else self._mesh.dy_cell

    def _scale_open_copies(
        self, u: np.ndarray, v: np.ndarray, open_faces: dict
    ) -> None:
        """Arm F: scale the open scaled-rule faces so the total outflow equals the supply.

        The outflow of the fixed segments is subtracted from the supply
        first. When the open copies carry no outflow (at rest, or all
        inward) the open faces take the equal-velocity split instead, and
        the iteration is counted as a fallback. The scaled faces then leave
        the outlet masks, so the correction holds them.
        """
        fixed_out = 0.0
        open_out = 0.0
        open_width = 0.0
        for seg in self.segments:
            face = face_row(u, v, seg.edge)
            widths = self._widths(seg.edge)
            if seg.rule == "fixed":
                fixed_out += seg.speed * float(widths[seg.mask].sum())
            elif seg.rule == "scaled":
                m = seg.mask & open_faces[seg.edge]
                out = -face[m] if seg.edge == "bottom" else face[m]
                open_out += float(np.sum(out * widths[m]))
                open_width += float(widths[m].sum())
        target = self._supply - fixed_out
        if open_width == 0.0:
            return
        if open_out > 0.0:
            scale = target / open_out
            self.scale_history.append(scale)
        else:
            scale = float("nan")
            self.fallback_iterations += 1
        for seg in self.segments:
            if seg.rule != "scaled":
                continue
            face = face_row(u, v, seg.edge)
            m = seg.mask & open_faces[seg.edge]
            if open_out > 0.0:
                face[m] *= scale
            else:
                face[m] = outward_value(target / open_width, seg.edge)
            open_faces[seg.edge] &= ~m

    def _extrapolate_outlets(self, u: np.ndarray, v: np.ndarray) -> None:
        open_faces = {edge: self._base[edge].copy() for edge in self._base}
        closed = {edge: np.zeros_like(self._base[edge]) for edge in self._base}
        for seg in self.segments:
            face = face_row(u, v, seg.edge)
            m = seg.mask
            if seg.rule == "fixed":
                face[m] = outward_value(seg.speed, seg.edge)
                open_faces[seg.edge] &= ~m
                continue
            interior = interior_row(u, v, seg.edge)
            if seg.rule in ("copy", "scaled", "copy_open"):
                candidate = interior[m].copy()
            elif seg.rule == "local":
                candidate = closing_velocity(u, v, seg.edge, self._mesh)[m]
            elif seg.rule == "stick":
                if self.iteration == 0:
                    candidate = interior[m].copy()
                else:
                    candidate = face[m].copy()
                    reopen = self.closed[seg.edge][m]
                    candidate[reopen] = interior[m][reopen]
            else:
                raise ValueError(f"unknown rule {seg.rule!r}")
            face[m] = candidate
            if seg.rule == "copy_open":
                # A-open: an inward copy stays open and is corrected like any
                # other outlet face.
                continue
            reversed_faces = np.zeros_like(m)
            reversed_faces[m] = inward(candidate, seg.edge)
            face[reversed_faces] = 0.0
            open_faces[seg.edge] &= ~reversed_faces
            closed[seg.edge] |= reversed_faces
        if any(seg.rule == "scaled" for seg in self.segments):
            self._scale_open_copies(u, v, open_faces)
        self.closed = closed
        self.open = open_faces
        corrector = self._corrector
        corrector._out_bottom = open_faces["bottom"]
        corrector._out_right = open_faces["right"]
        any_open = bool(open_faces["bottom"].any() or open_faces["right"].any())
        corrector.needs_pin = not any_open
        corrector.pin_cell = self._pin
        if not any_open:
            self.pinned_iterations += 1
        self.iteration += 1


# ---------------------------------------------------------------------------
# Rooms
# ---------------------------------------------------------------------------


@dataclass
class Room:
    """A built solver and what the runner needs beside it."""

    name: str
    cfg: SimConfig
    mesh: Mesh
    boundary: StaggeredBoundary
    solver: StaggeredSolver
    segments: list[Segment]
    meta: dict


def stopping_for(
    nx: int, ny: int, width: float, height: float, rho: float, t_end: float
) -> dict:
    """error_estimate with the defaults' error tolerance and ADR-011 G's per-cell bound."""
    v_min = (width / nx) * (height / ny)
    return {
        "stopping_rule": "error_estimate",
        "iteration_error_tol": 1e-6,
        "mass_imbalance_tol": 1e-4 * rho * v_min / t_end,
    }


def product_raw(
    nx: int, ny: int, mu_factor: float, n_outer: int, rtol: float, stopping: dict | None
) -> dict:
    """configs/clean_room_default.yaml regridded, viscosity scaled, the ladder's solver keys."""
    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    raw["fluid"]["viscosity"] *= mu_factor
    block = raw["solver"]
    block["max_simple_iter"] = n_outer
    block["alpha_velocity"] = 0.5
    block["max_pressure_iter"] = 5000
    block["pressure_rtol"] = rtol
    if stopping:
        block.update(stopping)
    return raw


def product_segments(
    mesh: Mesh, cfg: SimConfig, boundary: StaggeredBoundary, arm: str
) -> tuple[list[Segment], dict]:
    """The four floor returns and the hood under one arm, with the flow split recorded."""
    bottom_names = segment_names(mesh, cfg, "bottom")
    right_names = segment_names(mesh, cfg, "right")
    returns = sorted({n for n in bottom_names if n.startswith("floor_return")})
    masks = {n: bottom_names == n for n in returns}
    hood = right_names == "hood_exhaust"
    supply = boundary.get_total_inlet_flux()
    hood_flux = HOOD_SPEED * float(mesh.dy_cell[hood].sum())
    widths = {n: float(mesh.dx_cell[m].sum()) for n, m in masks.items()}
    total_width = sum(widths.values())
    fixed_speed = (supply - hood_flux) / total_width
    largest = max(returns, key=lambda n: widths[n])
    segments = []
    for n in returns:
        rule = RULES[arm]
        if arm in ("E", "E0") and n != largest:
            rule = "fixed"
        speed = fixed_speed if rule == "fixed" else 0.0
        segments.append(Segment(n, "bottom", masks[n], rule, speed))
    segments.append(Segment("hood_exhaust", "right", hood, "fixed", HOOD_SPEED))
    split = {
        "supply_flux": supply,
        "hood_flux": hood_flux,
        "hood_faces": int(hood.sum()),
        "return_faces": {n: int(m.sum()) for n, m in masks.items()},
        "return_widths": widths,
        "fixed_speed": fixed_speed,
        "largest_return": largest,
        "rules": {s.name: s.rule for s in segments},
    }
    return segments, split


def product_room(
    name: str,
    arm: str,
    nx: int,
    ny: int,
    mu_factor: float,
    n_outer: int,
    rtol: float,
    sweeps: int,
    stopping: dict | None,
    hood_gradient: bool,
) -> Room:
    """The product room under one arm.

    The hood's tangential velocity is held at zero unless hood_gradient
    keeps the pressure outlet's zero gradient (control A0). Under D0 the
    floor returns' tangential velocity is held at zero as well; under every
    other arm it stays the pressure outlet's zero gradient, as the committed
    boundary layer gives it.
    """
    raw = product_raw(nx, ny, mu_factor, n_outer, rtol, stopping)
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    switched = 0 if hood_gradient else hold_hood_tangential(boundary)
    returns_switched = hold_tangential(boundary, "bottom") if arm == "D0" else 0
    segments, split = product_segments(mesh, cfg, boundary, arm)
    solver = ArmSolver(mesh, cfg, boundary, segments, sweeps)
    meta = {
        "case": "product",
        "arm": arm,
        "nx": nx,
        "ny": ny,
        "mu_factor": mu_factor,
        "mu": cfg.mu,
        "reynolds": cfg.rho * 0.45 * cfg.room_height / cfg.mu,
        "sweeps": sweeps,
        "alpha_velocity": cfg.alpha_velocity,
        "alpha_pressure": cfg.alpha_pressure,
        "pressure_rtol": rtol,
        "max_pressure_iter": cfg.max_pressure_iter,
        "stopping": stopping or {"stopping_rule": "velocity_step"},
        "convergence_tol": cfg.convergence_tol,
        "hood_tangential": "zero_gradient" if hood_gradient else "zero",
        "hood_tangential_switched": switched,
        "return_tangential": "zero" if arm == "D0" else "zero_gradient",
        "return_tangential_switched": returns_switched,
        "split": split,
    }
    return Room(name, cfg, mesh, boundary, solver, segments, meta)


def val001_room(name: str, arm: str, grid: tuple[int, int] | None = None) -> Room:
    """VAL-001 as the harness loads it, under the committed solver or arm B, C or F.

    The preset is val001_80x40; with grid the case file is loaded on that
    grid instead (validation.cases.load_case), every other key unchanged.
    """
    cfg = load_preset("val001_80x40") if grid is None else load_case("poiseuille", grid)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    right_names = segment_names(mesh, cfg, "right")
    outlet = right_names == "outlet"
    if arm == "committed":
        solver = StaggeredSolver(mesh, cfg, boundary)
        solver._corrector = RecordingCorrector(mesh, cfg, boundary)
        segments = [Segment("outlet", "right", outlet, "copy")]
    else:
        segments = [Segment("outlet", "right", outlet, RULES[arm])]
        solver = ArmSolver(mesh, cfg, boundary, segments, 1)
    meta = {
        "case": f"val001_{cfg.nx}x{cfg.ny}",
        "arm": arm,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "alpha_velocity": cfg.alpha_velocity,
        "pressure_rtol": cfg.pressure_rtol,
        "stopping": {
            "stopping_rule": cfg.stopping_rule,
            "iteration_error_tol": cfg.iteration_error_tol,
            "mass_imbalance_tol": cfg.mass_imbalance_tol,
        },
        "max_simple_iter": cfg.max_simple_iter,
    }
    return Room(name, cfg, mesh, boundary, solver, segments, meta)


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def face_hash(u: np.ndarray, v: np.ndarray) -> str:
    """Section 2.3 of docs/reports/ecr003_step2_baseline.md: SHA-256 over u's bytes then v's."""
    return hashlib.sha256(
        np.ascontiguousarray(u, dtype="<f8").tobytes()
        + np.ascontiguousarray(v, dtype="<f8").tobytes()
    ).hexdigest()


def run(room: Room, log_every: int = 100) -> dict:
    """Run a room to its stop, divergence or cap; write NAME.json and NAME.npz."""
    OUT.mkdir(parents=True, exist_ok=True)
    solver = room.solver
    corrector: RecordingCorrector = solver._corrector  # type: ignore[assignment]
    mesh = room.mesh
    not_solid = mesh.cell_type != SOLID
    xc, yc = np.meshgrid(mesh.xc, mesh.yc)
    tol = room.cfg.convergence_tol
    started = datetime.now().isoformat(timespec="seconds")
    print(f"{room.name} start {started} {json.dumps(room.meta)}", flush=True)
    keys = (
        "residual",
        "p_mean",
        "b_norm",
        "b_share",
        "inner",
        "cap",
        "pinned",
        "max_speed",
        "at",
        "worst",
        "signed",
        "reversed",
        "closed",
    )
    rec: dict[str, list] = {k: [] for k in keys}
    vs_stop: dict[str, int] = {}
    last: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()

    def callback(state: IterationState) -> None:
        speed = np.hypot(state.u, state.v)
        k = int(np.argmax(speed))
        top = float(speed.flat[k])
        u, v = corrector.last_u, corrector.last_v
        assert u is not None and v is not None
        imbalance = corrector.mass_imbalance(u, v)
        record = corrector.records[-1]
        reversed_faces, closed_faces = [], []
        closed = getattr(solver, "closed", None)
        for seg in room.segments:
            values = face_row(u, v, seg.edge)[seg.mask]
            reversed_faces.append(int(np.sum(inward(values, seg.edge))))
            closed_faces.append(
                int(closed[seg.edge][seg.mask].sum()) if closed is not None else 0
            )
        rec["residual"].append(float(state.residual))
        rec["p_mean"].append(float(np.mean(state.p[not_solid])))
        rec["b_norm"].append(record.b_norm)
        rec["b_share"].append(record.b_share_open)
        rec["inner"].append(int(record.iterations))
        rec["cap"].append(bool(record.reached_cap))
        rec["pinned"].append(bool(record.needs_pin))
        rec["max_speed"].append(top)
        rec["at"].append((round(float(xc.flat[k]), 3), round(float(yc.flat[k]), 3)))
        rec["worst"].append(float(np.max(np.abs(imbalance))))
        rec["signed"].append(float(np.sum(imbalance)))
        rec["reversed"].append(reversed_faces)
        rec["closed"].append(closed_faces)
        last["u_c"], last["v_c"], last["p"] = (
            state.u.copy(),
            state.v.copy(),
            state.p.copy(),
        )
        if (
            "velocity_step" not in vs_stop
            and state.residual < tol
            and not record.reached_cap
        ):
            # 1-based, as outer37.py recorded it (233 on the drift case).
            vs_stop["velocity_step"] = state.iteration + 1
        it = state.iteration
        if it % log_every == 0:
            print(
                f"{room.name} it {it:5d} res {state.residual:.3e} inner {record.iterations:5d} "
                f"|b| {record.b_norm:.3e} share {record.b_share_open:.3f} "
                f"p_mean {rec['p_mean'][-1]:+.4f} max|U| {top:.4g} at {rec['at'][-1]} "
                f"rev {reversed_faces} closed {closed_faces} t {time.perf_counter() - t0:7.1f}s",
                flush=True,
            )
        if not math.isfinite(top) or top > DIVERGED_SPEED:
            raise DivergedError

    try:
        solver.solve_steady(on_iteration=callback)
        stop = solver.stop_reason
    except DivergedError:
        stop = "diverged"
    n = len(rec["residual"])
    p_mean = np.array(rec["p_mean"])
    window = min(100, n - 1)
    drift = (
        float(np.mean(np.diff(p_mean[-(window + 1) :]))) if window > 0 else float("nan")
    )
    u_f, v_f = corrector.last_u, corrector.last_v
    assert u_f is not None and v_f is not None
    faces = solver.face_velocities
    if faces is None:
        # A diverged run leaves the loop before the solver keeps its faces;
        # the last correction's are the ones to hash.
        faces = FaceVelocities.copy_of(u_f, v_f)
    out = {
        "name": room.name,
        "started": started,
        "seconds": time.perf_counter() - t0,
        **room.meta,
        "segments": [s.name for s in room.segments],
        "faces_per_segment": [int(s.mask.sum()) for s in room.segments],
        "stop": stop,
        "outer": n,
        "velocity_step_outer": vs_stop.get("velocity_step"),
        "p_drift_last100": drift,
        "p_drift_window": window,
        "b_norm_end": rec["b_norm"][-1],
        "b_share_end": rec["b_share"][-1],
        "worst_end": rec["worst"][-1],
        "signed_end": rec["signed"][-1],
        "residual_min": float(np.min(rec["residual"])),
        "residual_end": rec["residual"][-1],
        "max_speed_end": rec["max_speed"][-1],
        "at_end": rec["at"][-1],
        "reversed_most": [int(x) for x in np.max(np.array(rec["reversed"]), axis=0)],
        "closed_most": [int(x) for x in np.max(np.array(rec["closed"]), axis=0)],
        "reversed_end": rec["reversed"][-1],
        "closed_end": rec["closed"][-1],
        "reversed_any_iterations": int(
            np.sum(np.array(rec["reversed"]).sum(axis=1) > 0)
        ),
        "closed_any_iterations": int(np.sum(np.array(rec["closed"]).sum(axis=1) > 0)),
        "cap_hits": int(sum(rec["cap"])),
        "pinned_iterations": int(sum(rec["pinned"])),
        "fallback_iterations": getattr(solver, "fallback_iterations", 0),
        "scale_end": (getattr(solver, "scale_history", [None]) or [None])[-1],
        "scale_extremes": (
            [float(min(solver.scale_history)), float(max(solver.scale_history))]
            if getattr(solver, "scale_history", [])
            else None
        ),
        "inner_median": float(np.median(rec["inner"])),
        "inner_max": int(np.max(rec["inner"])),
        "face_hash": face_hash(faces.u, faces.v),
        "faces_match_last_correction": bool(
            np.array_equal(faces.u, u_f) and np.array_equal(faces.v, v_f)
        ),
        **rec,
    }
    if room.meta["case"] == "val001_80x40":
        out["metric"] = poiseuille_l2_error(room.cfg, mesh, last["u_c"]).as_dict()
    (OUT / f"{room.name}.json").write_text(json.dumps(out))
    np.savez(
        OUT / f"{room.name}.npz",
        u_faces=faces.u,
        v_faces=faces.v,
        p=last["p"],
        u_c=last["u_c"],
        v_c=last["v_c"],
        last_b=corrector.last_b,
    )
    print(
        f"{room.name} done {datetime.now().isoformat(timespec='seconds')} stop {stop} after {n} "
        f"(velocity_step at {out['velocity_step_outer']}); drift {drift:+.3e} Pa/outer; "
        f"|b| end {out['b_norm_end']:.3e} share {out['b_share_end']:.3f}; worst {out['worst_end']:.2e} "
        f"signed {out['signed_end']:+.2e}; rev most {out['reversed_most']} closed most "
        f"{out['closed_most']}; cap hits {out['cap_hits']}; {out['seconds']:.0f} s",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def drift(args: argparse.Namespace) -> None:
    """Measurement 1: the 80x30 drift case under one arm at one pressure_rtol."""
    rtol = float(args.rtol)
    tag = args.tag or f"drift_{args.arm}_{args.rtol}" + (
        "_gradient" if args.hood_gradient else ""
    )
    stopping = stopping_for(80, 30, 8.0, 3.0, 1.2, 60.0)
    if args.long:
        # Tolerances no run meets, so the loop runs to --n-outer and the
        # pressure's movement can be watched past the stop.
        stopping.update(iteration_error_tol=1e-14, mass_imbalance_tol=1e-16)
        tag += "_long"
    room = product_room(
        tag,
        args.arm,
        80,
        30,
        1000.0,
        args.n_outer,
        rtol,
        10,
        stopping,
        args.hood_gradient,
    )
    run(room)


def ladder(args: argparse.Namespace) -> None:
    """Measurement 2: one rung of the 40x15 ladder under one arm."""
    tag = f"ladder_{args.arm}_{args.rung}"
    room = product_room(
        tag,
        args.arm,
        40,
        15,
        LADDER_FACTORS[args.rung],
        args.n_outer,
        1e-8,
        1,
        None,
        False,
    )
    run(room, log_every=25)


def val001(args: argparse.Namespace) -> None:
    """Measurement 3: VAL-001 under the committed path or arm B, C or F."""
    grid = tuple(args.grid) if args.grid else None
    suffix = f"_{grid[0]}x{grid[1]}" if grid else ""
    run(val001_room(f"val001_{args.arm}{suffix}", args.arm, grid))


def control(args: argparse.Namespace) -> None:
    """SweepPredictor against frozen34.py's FrozenPredictor, zero field, ten sweeps, 50 outer."""
    sys.path.insert(0, str(ROOT / "results" / "builder34"))
    import frozen34

    histories = {}
    hashes = {}
    for label in ("sweep", "frozen"):
        room = product_room(
            f"control_{label}",
            "A",
            80,
            30,
            1000.0,
            50,
            1e-8,
            10,
            stopping_for(80, 30, 8.0, 3.0, 1.2, 60.0),
            False,
        )
        if label == "frozen":
            room.solver._predictor = frozen34.FrozenPredictor(
                room.mesh,
                room.cfg,
                room.boundary,
                np.zeros(room.mesh.cell_type.shape),
                sweeps=10,
            )
        out = run(room, log_every=10)
        histories[label] = out["residual"]
        hashes[label] = out["face_hash"]
    same = (
        histories["sweep"] == histories["frozen"]
        and hashes["sweep"] == hashes["frozen"]
    )
    result = {
        "residual_bitwise_equal": same,
        "hashes": hashes,
        "outer": len(histories["sweep"]),
    }
    (OUT / "control.json").write_text(json.dumps(result))
    print("control:", result, flush=True)


def compare(args: argparse.Namespace) -> None:
    """Measurement 4: the largest cell-centred difference between two runs.

    Differences, their argmax and the scales are taken over non-SOLID cells
    only (review 41 B1: the pressure is 0 in SOLID cells, so a difference
    there is the difference of the two means). The scale is the first-named
    run's, and the record says so.
    """
    a = np.load(OUT / f"{args.names[0]}.npz")
    b = np.load(OUT / f"{args.names[1]}.npz")
    nx, ny = args.grid
    cfg = SimConfig.from_dict(product_raw(nx, ny, 1.0, 1, 1e-8, None))
    mesh = Mesh(cfg)
    fluid = mesh.cell_type != SOLID
    xc, yc = np.meshgrid(mesh.xc, mesh.yc)
    out: dict = {"names": list(args.names), "scale_of": args.names[0]}

    def largest(diff: np.ndarray, scale_field: np.ndarray) -> dict:
        masked = np.where(fluid, diff, 0.0)
        k = int(np.argmax(masked))
        return {
            "max_abs_diff": float(masked.flat[k]),
            "at": (round(float(xc.flat[k]), 3), round(float(yc.flat[k]), 3)),
            "scale": float(np.max(np.abs(scale_field[fluid]))),
        }

    for comp in ("u_c", "v_c"):
        out[comp] = largest(np.abs(a[comp] - b[comp]), a[comp])
    pa = a["p"] - np.mean(a["p"][fluid])
    pb = b["p"] - np.mean(b["p"][fluid])
    out["p_demeaned"] = largest(np.abs(pa - pb), pa)
    for comp, edge, row in (("v_faces", "bottom", 0), ("u_faces", "right", -1)):
        fa = a[comp][row, :] if edge == "bottom" else a[comp][:, row]
        fb = b[comp][row, :] if edge == "bottom" else b[comp][:, row]
        out[f"{edge}_faces_max_abs_diff"] = float(np.max(np.abs(fa - fb)))
    name = f"compare_{args.names[0]}_vs_{args.names[1]}"
    (OUT / f"{name}.json").write_text(json.dumps(out))
    print(name, json.dumps(out), flush=True)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("drift")
    p.add_argument("arm", choices=sorted(RULES))
    p.add_argument("rtol")
    p.add_argument("--n-outer", type=int, default=20000)
    p.add_argument("--hood-gradient", action="store_true")
    p.add_argument("--long", action="store_true")
    p.add_argument("--tag", default=None)
    p.set_defaults(func=drift)
    p = sub.add_parser("ladder")
    p.add_argument("arm", choices=sorted(RULES))
    p.add_argument("rung", choices=sorted(LADDER_FACTORS))
    p.add_argument("--n-outer", type=int, default=3000)
    p.set_defaults(func=ladder)
    p = sub.add_parser("val001")
    p.add_argument("arm", choices=["committed", "B", "C", "F"])
    p.add_argument("--grid", type=int, nargs=2, default=None)
    p.set_defaults(func=val001)
    p = sub.add_parser("control")
    p.set_defaults(func=control)
    p = sub.add_parser("compare")
    p.add_argument("names", nargs=2)
    p.add_argument("--grid", type=int, nargs=2, default=(80, 30))
    p.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
