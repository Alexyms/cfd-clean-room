"""Boundary conditions imposed directly on the staggered (MAC) fields (REQ-S12).

On the layout of src/staggered.py the normal velocity component at every
domain edge is a storage location: ``u[:, 0]`` and ``u[:, nx]`` are the
left and right faces, ``v[0, :]`` and ``v[ny, :]`` the bottom and top
faces. A Dirichlet condition on the normal component is therefore a plain
assignment, exact, with no interpolation and no error term. That assignment
is the first thing this module provides (``apply_normal_velocity``).

The tangential component is not a storage location. Along the bottom wall
``u`` lives at cell-center height, ``dy_face[0]`` above the wall, and
no-slip asks for ``u = 0`` at the wall itself, between storage rows. There
is no entry to write. Writing a mirrored value into a row outside the
domain so that an interpolation comes out right would be a ghost cell under
another name, carrying the O(h) error ECR-001 exists to remove. So the
second thing this module provides is data (``tangential_conditions``): for
every tangential storage location along an edge, whether a value is
prescribed, what it is, and how far the wall is from the storage row. The
momentum step (ECR-001 step 4) forms the one-sided wall gradient
``(u[0, i] - u_wall) / dy_face[0]`` from that data. Pressure is data only
as well (``pressure_outlets``): which edge cells are outlets and the gauge
value there. A wall needs no pressure condition on a staggered grid; the
homogeneous Neumann condition falls out of omitting the wall face from the
divergence and gradient operators, which is step 5's work.

Nothing here reads or writes a location outside the domain. The arrays the
imposer touches have exactly the shapes src/staggered.py allocates, and
every index it writes is a domain face.

``mesh.cell_type`` is unchanged by ECR-001 step 3: the BOUNDARY ring stays
so both solvers run on one mesh until step 6 retires the collocated solver.
On the staggered grid a BOUNDARY cell is an ordinary cell that touches an
edge. A SOLID cell on an edge is a wall whatever segment the configuration
puts there, because the obstacle is what bounds the flow at that face. Which
segment covers each edge cell, with that rule applied, is read from the
registry's ``coverage_along`` on the coordinates and SOLID mask
``staggered.edge_cell_inputs`` derives, the same call the concentration
layer makes (REQ-S12.1); nothing here decides coverage or its inputs on its
own, and the inlet flux is attributed to a segment by the name that
coverage gives each face.

A ``fixed_flow_outlet`` is an inlet run backwards (ADR-012 D, amended
2026-10-08): its faces hold an outward normal velocity as a Dirichlet value
and a zero tangential velocity, and it is neither a pressure outlet nor part
of the inlet flux. A segment states its velocity or states none. Those that
state none share, at one face velocity, what the stated outlets leave of the
discrete inflow (``fixed_flow_velocities``), so outflow equals inflow to
rounding on any mesh, however the faces round each opening. A domain whose
only outlets are fixed-flow therefore has no pressure outlet and the pressure
correction takes its closed-domain path.

Which edge faces are walls is data too (``wall_faces``, ECR-002 step 6): a
face with zero normal velocity that is not an outlet, at rest or moving,
which is where the wall functions of ADR-012 B apply. An inlet that admits
air and either kind of outlet is not a wall.

Equipment vents (ADR-014 J, step 0b). A segment on an obstacle face is a
cooling discharge (``velocity_inlet``) or a process exhaust
(``fixed_flow_outlet``). Its faces are interior entries of the face arrays,
the faces between the obstacle's cells and the room that
``staggered.obstacle_face_inputs`` finds and the registry's ``coverage_on``
gives to the vent (``vent_runs``, the one derivation every vent layer
reads). ``apply_normal_velocity`` writes their normal velocity once, as it
writes an inlet's; the inflow sums count a discharge and the remainder rule
counts an exhaust as a stated outlet, so the returns that state no velocity
share what the supply leaves after the hood and every exhaust. A file whose
outlets all state their velocities, with no pressure outlet, must balance on
the mesh's faces as it balanced on the configured extents at load. Which
faces are vents, normal and tangential, is data (``vent_faces``) for the
momentum, turbulence and concentration layers.
"""

from dataclasses import dataclass

import numpy as np

from src.boundary_registry import (
    EDGES,
    FIXED_FLOW_OUTLET,
    OBSTACLE,
    PRESSURE_OUTLET,
    VELOCITY_INLET,
    WALL,
    BoundaryRegistry,
    EdgeCondition,
    EdgeCoverage,
    fixed_flow_condition,
)
from src.config import BoundarySpec, SimConfig
from src.mesh import Mesh
from src.staggered import (
    ROOM_EDGE_OF_FACE,
    ObstacleFace,
    edge_cell_inputs,
    obstacle_face_inputs,
    u_shape,
    v_shape,
)

# Sign that turns the prescribed normal component into flow into the domain.
_INWARD_SIGN: dict[str, float] = {
    "bottom": 1.0,
    "top": -1.0,
    "left": 1.0,
    "right": -1.0,
}


# The relative difference between the inflow and the stated outflow on the
# mesh's faces below which a room no outlet balances is read as balanced: the
# rounding of two face sums, not a tolerance a case chooses.
MESH_BALANCE_TOLERANCE = 1.0e-12


@dataclass(frozen=True, eq=False)
class VentRun:
    """The mesh faces one vent segment covers (ADR-014 J).

    Parameters
    ----------
    name : str
        The segment's name.
    spec : BoundarySpec
        The segment.
    edge : str
        The domain edge the obstacle face plays to the room
        (``staggered.ROOM_EDGE_OF_FACE``): "bottom" for a top face.
    component : str
        "v" on a top or bottom face, "u" on a side.
    rows, cols : np.ndarray
        int64, the vent's faces in its component's array, in order along
        the face.
    widths : np.ndarray
        Each face's width, metres.
    condition : EdgeCondition
        The vent's condition on ``edge``: the normal component signed along
        its axis, zero tangential.
    corner_rows, corner_cols : np.ndarray
        int64, the tangential wall-stencil faces over the vent's span, in
        ``wall_mu``'s [ny+1, nx+1] layout (``staggered.ObstacleFace``).
    """

    name: str
    spec: BoundarySpec
    edge: str
    component: str
    rows: np.ndarray
    cols: np.ndarray
    widths: np.ndarray
    condition: EdgeCondition
    corner_rows: np.ndarray
    corner_cols: np.ndarray

    @property
    def normal(self) -> float:
        """The normal velocity stored on every face, m/s, signed along the axis."""
        if self.component == "v":
            return self.condition.v_prescribed
        return self.condition.u_prescribed

    @property
    def is_inlet(self) -> bool:
        """True for a cooling discharge, False for a process exhaust."""
        return self.condition.bc_type == VELOCITY_INLET


@dataclass(frozen=True, eq=False)
class VentFaces:
    """Which faces are vents, for the layers that treat them apart (ADR-014 J).

    All arrays are boolean and read-only; every one is all False in a
    configuration without vents.

    Parameters
    ----------
    normal_u : np.ndarray
        [ny, nx+1]: the u faces of a vent on an obstacle side.
    normal_v : np.ndarray
        [ny+1, nx]: the v faces of a vent on an obstacle top or bottom.
    tangential_u : np.ndarray
        [ny+1, nx+1], ``wall_mu``'s "u" layout: the horizontal faces of the
        u wall stencil over a top or bottom vent's span.
    tangential_v : np.ndarray
        [ny+1, nx+1], ``wall_mu``'s "v" layout: the vertical faces of the v
        wall stencil over a side vent's span.
    """

    normal_u: np.ndarray
    normal_v: np.ndarray
    tangential_u: np.ndarray
    tangential_v: np.ndarray

    @property
    def any(self) -> bool:
        """True when the configuration has a vent face."""
        return bool(self.normal_u.any() or self.normal_v.any())


def vent_runs(
    mesh: Mesh, config: SimConfig, registry: BoundaryRegistry
) -> list[VentRun]:
    """The faces of every vent segment on the mesh, in configuration order (ADR-014 J).

    The one derivation of a vent's faces: ``staggered.obstacle_face_inputs``
    for the faces of each obstacle face a vent names, and the registry's
    ``coverage_on`` for which vent covers each. The velocity, turbulence and
    concentration layers all call it.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    config : SimConfig
        Supplies the obstacles.
    registry : BoundaryRegistry
        The configured segments.

    Returns
    -------
    list[VentRun]
        One per vent segment; empty without vents.

    Raises
    ------
    ValueError
        If a vent's extent covers no face of this mesh (ADR-014 J's load
        rule that needs the mesh).
    """
    vents = registry.vents()
    if not vents:
        return []
    obstacles = {obs.name: obs for obs in config.obstacles}
    faces: dict[tuple[str, str], tuple[ObstacleFace, list, list]] = {}
    runs: list[VentRun] = []
    for name, spec in vents.items():
        assert spec.obstacle is not None and spec.face is not None
        key = (spec.obstacle, spec.face)
        if key not in faces:
            inputs = obstacle_face_inputs(mesh, obstacles[spec.obstacle], spec.face)
            faces[key] = (
                inputs,
                registry.coverage_on(spec.obstacle, spec.face, inputs.coordinates),
                registry.coverage_on(
                    spec.obstacle, spec.face, inputs.corner_coordinates
                ),
            )
        inputs, coverage, corners = faces[key]
        mine = np.array([point.name == name for point in coverage], dtype=bool)
        if not mine.any():
            raise ValueError(
                f"boundaries.{name}: the vent's extent covers no face of the "
                f"{spec.face} face of '{spec.obstacle}' on this mesh; widen it to "
                "hold at least one cell centre of the face"
            )
        mine_corner = np.array([point.name == name for point in corners], dtype=bool)
        condition = coverage[int(np.argmax(mine))].condition
        runs.append(
            VentRun(
                name=name,
                spec=spec,
                edge=ROOM_EDGE_OF_FACE[spec.face],
                component=inputs.component,
                rows=inputs.rows[mine],
                cols=inputs.cols[mine],
                widths=inputs.widths[mine],
                condition=condition,
                corner_rows=inputs.corner_rows[mine_corner],
                corner_cols=inputs.corner_cols[mine_corner],
            )
        )
    return runs


def vent_face_masks(mesh: Mesh, runs: list[VentRun]) -> VentFaces:
    """The vent masks of a list of runs (``StaggeredBoundary.vent_faces``).

    Parameters
    ----------
    mesh : Mesh
        Fixes the shapes.
    runs : list[VentRun]
        From ``vent_runs``.

    Returns
    -------
    VentFaces
        Read-only masks; all False for an empty list.
    """
    ny, nxp1 = u_shape(mesh)
    normal_u = np.zeros(u_shape(mesh), dtype=bool)
    normal_v = np.zeros(v_shape(mesh), dtype=bool)
    tangential_u = np.zeros((ny + 1, nxp1), dtype=bool)
    tangential_v = np.zeros((ny + 1, nxp1), dtype=bool)
    for run in runs:
        if run.component == "v":
            normal_v[run.rows, run.cols] = True
            tangential_u[run.corner_rows, run.corner_cols] = True
        else:
            normal_u[run.rows, run.cols] = True
            tangential_v[run.corner_rows, run.corner_cols] = True
    for array in (normal_u, normal_v, tangential_u, tangential_v):
        array.flags.writeable = False
    return VentFaces(normal_u, normal_v, tangential_u, tangential_v)


# eq=False because the default __eq__ compares the array fields elementwise
# and raises on the ambiguous truth value (issue 51 D2); identity is the only
# comparison that makes sense for data built once from the mesh.
@dataclass(frozen=True, eq=False)
class TangentialCondition:
    """Dirichlet data for the tangential velocity along one domain edge.

    One entry per tangential storage location along the edge: the ``u``
    columns at ``x[0..nx]`` on the bottom and top edges, the ``v`` rows at
    ``y[0..ny]`` on the left and right edges. The two end entries coincide
    with normal storage locations of the adjacent edges, which the imposer
    sets exactly; they are carried here so the arrays index like the field.

    Parameters
    ----------
    edge : str
        One of "bottom", "top", "left", "right".
    component : str
        "u" on the bottom and top edges, "v" on the left and right edges.
    is_dirichlet : np.ndarray
        Boolean, shape [nx+1] or [ny+1]. True where the wall value is
        prescribed (wall, velocity_inlet, fixed_flow_outlet, or a face of a
        SOLID edge cell); False at pressure outlets, where the component
        has zero gradient.
    value : np.ndarray
        Prescribed tangential velocity at the wall, same shape. Zero
        wherever ``is_dirichlet`` is False.
    wall_distance : float
        Distance from the wall to the storage row or column: ``dy_face[0]``,
        ``dy_face[ny]``, ``dx_face[0]`` or ``dx_face[nx]``. Read from the
        mesh, so it is the geometric distance on a stretched mesh too.
    """

    edge: str
    component: str
    is_dirichlet: np.ndarray
    value: np.ndarray
    wall_distance: float


@dataclass(frozen=True, eq=False)
class EdgeWalls:
    """Which faces along one domain edge are walls (ECR-002 step 6).

    A wall is a face with zero normal velocity that is not an outlet: a
    ``wall`` segment, a point no segment covers, or a ``velocity_inlet``
    whose normal component is zero (a moving wall). An inlet that admits
    air, a ``pressure_outlet`` and a ``fixed_flow_outlet`` are not walls.

    Parameters
    ----------
    edge : str
        One of "bottom", "top", "left", "right".
    cells : np.ndarray
        Boolean, [nx] on the bottom and top edges, [ny] on the left and
        right edges: True where the edge cell's own face is a wall and the
        cell is not SOLID (a SOLID cell has no face toward the flow).
    corners : np.ndarray
        Boolean, [nx+1] or [ny+1], at the tangential storage locations of
        ``TangentialCondition``, from the same coverage: True where the
        condition there is a wall, a point bounding a SOLID edge cell
        included.
    """

    edge: str
    cells: np.ndarray
    corners: np.ndarray


@dataclass(frozen=True)
class PressureOutletCondition:
    """Which cells along one domain edge have a pressure outlet face.

    Parameters
    ----------
    edge : str
        One of "bottom", "top", "left", "right".
    is_outlet : np.ndarray
        Boolean, shape [nx] on the bottom and top edges, [ny] on the left
        and right edges. True where the cell's edge face is an outlet.
    pressure : float
        Pressure at the outlet face. The outlet is the gauge datum, so it
        is zero by definition; BoundarySpec carries no pressure value.
    """

    edge: str
    is_outlet: np.ndarray
    pressure: float


class StaggeredBoundary:
    """Impose Dirichlet normal velocities and expose the remaining conditions as data.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh. Supplies the edge coordinates, the face
        widths, the wall-to-first-center distances and ``cell_type``.
    config : SimConfig
        Simulation configuration with boundary definitions.

    Raises
    ------
    ValueError
        If the fixed-flow outlets that state no velocity cover no face, or
        the inflow the stated ones leave for them is zero or negative; if a
        vent covers no face of the mesh; or if every outlet states its
        velocity, none is a pressure outlet, and the stated outflow differs
        from the inflow on the mesh's faces.
    """

    def __init__(self, mesh: Mesh, config: SimConfig) -> None:
        self._mesh = mesh
        self._registry = BoundaryRegistry(config)
        self._u_shape = u_shape(mesh)
        self._v_shape = v_shape(mesh)
        self._vents: list[VentRun] = vent_runs(mesh, config, self._registry)
        self._vent_faces: VentFaces = vent_face_masks(mesh, self._vents)

        # Coverage at the cell positions along each edge, from the shared
        # derivation so every query below and the scalar layer see one answer.
        self._cell_coverage: dict[str, list[EdgeCoverage]] = {
            edge: self._registry.coverage_along(edge, *edge_cell_inputs(mesh, edge))
            for edge in EDGES
        }
        self._fixed_flow: dict[str, float] = self._resolve_fixed_flows()
        self._cell_conditions: dict[str, list[EdgeCondition]] = {
            edge: [self._resolved_condition(point, edge) for point in coverage]
            for edge, coverage in self._cell_coverage.items()
        }

        self._normal_dirichlet: dict[str, np.ndarray] = {}
        self._normal_value: dict[str, np.ndarray] = {}
        for edge, conditions in self._cell_conditions.items():
            self._normal_dirichlet[edge] = np.array(
                [c.bc_type != PRESSURE_OUTLET for c in conditions], dtype=bool
            )
            self._normal_value[edge] = np.array(
                [self._normal_component(c, edge) for c in conditions],
                dtype=np.float64,
            )

        self._tangential = {edge: self._build_tangential(edge) for edge in EDGES}
        self._outlets = {edge: self._build_outlet(edge) for edge in EDGES}
        self._walls = {edge: self._build_walls(edge) for edge in EDGES}

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _face_coordinates(self, edge: str) -> np.ndarray:
        """Tangential storage coordinates along an edge: x on top/bottom, y on left/right."""
        if edge in ("top", "bottom"):
            return self._mesh.x
        return self._mesh.y

    def _face_widths(self, edge: str) -> np.ndarray:
        """Width of each edge face: dx_cell on top/bottom, dy_cell on left/right."""
        if edge in ("top", "bottom"):
            return self._mesh.dx_cell
        return self._mesh.dy_cell

    def _wall_distance(self, edge: str) -> float:
        """Distance from the edge to the first tangential storage row or column."""
        if edge == "bottom":
            return float(self._mesh.dy_face[0])
        if edge == "top":
            return float(self._mesh.dy_face[-1])
        if edge == "left":
            return float(self._mesh.dx_face[0])
        return float(self._mesh.dx_face[-1])

    def _solid_along(self, edge: str) -> np.ndarray:
        """Boolean per cell along an edge, True where that cell is SOLID."""
        return edge_cell_inputs(self._mesh, edge)[1]

    @staticmethod
    def _normal_component(condition: EdgeCondition, edge: str) -> float:
        """The prescribed component normal to an edge: v on top/bottom, u on left/right."""
        if edge in ("top", "bottom"):
            return condition.v_prescribed
        return condition.u_prescribed

    @staticmethod
    def _tangential_component(condition: EdgeCondition, edge: str) -> float:
        """The prescribed component tangential to an edge: u on top/bottom, v on left/right."""
        if edge in ("top", "bottom"):
            return condition.u_prescribed
        return condition.v_prescribed

    def _resolved_condition(self, point: EdgeCoverage, edge: str) -> EdgeCondition:
        """The coverage's condition, with a fixed-flow outlet's share filled in."""
        if point.condition.bc_type != FIXED_FLOW_OUTLET:
            return point.condition
        assert point.name is not None
        return fixed_flow_condition(edge, self._fixed_flow[point.name])

    def _resolve_fixed_flows(self) -> dict[str, float]:
        """The outward normal velocity of every fixed-flow outlet segment.

        A segment that states a velocity keeps it. The rest share the
        discrete inflow less the stated outlets' discrete outflow at one
        face velocity, over their covered face widths. The widths come from
        the faces the shared coverage gives each segment, so the balance
        holds on the mesh the solver runs, not on the configured lengths.

        Raises
        ------
        ValueError
            If a segment that states no velocity has no face to share the
            remainder over, or the remainder is zero or negative.
        """
        fixed = {
            name: spec
            for name, spec in self._registry.boundaries.items()
            if spec.type == FIXED_FLOW_OUTLET
        }
        lengths: dict[str, float] = {}
        vent_widths = {run.name: float(run.widths.sum()) for run in self._vents}
        for name, spec in fixed.items():
            if spec.location == OBSTACLE:
                lengths[name] = vent_widths[name]
                continue
            covered = np.array(
                [p.name == name for p in self._cell_coverage[spec.location]],
                dtype=bool,
            )
            lengths[name] = float(self._face_widths(spec.location)[covered].sum())
        resolved: dict[str, float] = {
            name: spec.velocity
            for name, spec in fixed.items()
            if spec.velocity is not None
        }
        shared = [name for name, spec in fixed.items() if spec.velocity is None]
        if not shared:
            self._check_stated_balance(resolved, lengths)
            return resolved
        inflow = self.get_total_inlet_flux()
        stated_outflow = sum(resolved[name] * lengths[name] for name in resolved)
        remainder = inflow - stated_outflow
        shared_length = sum(lengths[name] for name in shared)
        if shared_length <= 0.0:
            raise ValueError(
                f"fixed_flow_outlet segments {shared} state no velocity and cover "
                "no face of this mesh, so they cannot carry the remaining flow"
            )
        if remainder <= 0.0:
            raise ValueError(
                f"the stated fixed-flow outlets already carry the whole inflow: "
                f"inflow {inflow:.6g} m^2/s, stated outflow {stated_outflow:.6g} "
                f"m^2/s, remainder {remainder:.6g} m^2/s for {shared}, which must "
                "be positive"
            )
        for name in shared:
            resolved[name] = remainder / shared_length
        return {name: resolved[name] for name in fixed}

    def _check_stated_balance(
        self, resolved: dict[str, float], lengths: dict[str, float]
    ) -> None:
        """Raise unless stated outlets with no pressure outlet balance the inflow on the mesh.

        Load accepted the file on the configured extents (config.py); here
        the faces decide, since the mesh may round two openings differently.
        With a pressure outlet, or with no fixed-flow outlet, nothing is
        checked: the pressure outlet balances the room, and a room with no
        outlet at all is a closed cavity.
        """
        if not resolved or any(
            spec.type == PRESSURE_OUTLET for spec in self._registry.boundaries.values()
        ):
            return
        inflow = self.get_total_inlet_flux()
        outflow = sum(resolved[name] * lengths[name] for name in resolved)
        if abs(inflow - outflow) > MESH_BALANCE_TOLERANCE * max(abs(inflow), outflow):
            raise ValueError(
                f"every fixed-flow outlet {sorted(resolved)} states its velocity and "
                "nothing else balances the room, but on this mesh's faces the stated "
                f"outflow {outflow:.15g} m^2/s differs from the inflow {inflow:.15g} "
                "m^2/s; place the openings' ends on cell faces, or leave the velocity "
                "off a domain-edge outlet"
            )

    # ------------------------------------------------------------------
    # Construction of the data deliverables
    # ------------------------------------------------------------------

    def _corner_conditions(self, edge: str) -> list[EdgeCondition]:
        """The condition at every tangential storage location along an edge.

        The condition at a storage coordinate is the registry's answer at
        that coordinate, so at a point where two segments meet the first in
        configuration order wins, the same rule as everywhere else. A storage
        location that bounds a SOLID edge cell is a wall.
        """
        solid = self._solid_along(edge)
        solid_face = np.zeros(solid.shape[0] + 1, dtype=bool)
        solid_face[:-1] |= solid
        solid_face[1:] |= solid
        return [
            point.condition
            for point in self._registry.coverage_along(
                edge, self._face_coordinates(edge), solid_face
            )
        ]

    def _build_tangential(self, edge: str) -> TangentialCondition:
        """Tangential data at every storage location along an edge (``_corner_conditions``)."""
        conditions = self._corner_conditions(edge)
        is_dirichlet = np.array(
            [c.bc_type != PRESSURE_OUTLET for c in conditions], dtype=bool
        )
        value = np.array(
            [self._tangential_component(c, edge) for c in conditions],
            dtype=np.float64,
        )
        value[~is_dirichlet] = 0.0
        is_dirichlet.flags.writeable = False
        value.flags.writeable = False
        return TangentialCondition(
            edge=edge,
            component="u" if edge in ("top", "bottom") else "v",
            is_dirichlet=is_dirichlet,
            value=value,
            wall_distance=self._wall_distance(edge),
        )

    @staticmethod
    def _is_wall(condition: EdgeCondition, edge: str) -> bool:
        """Zero normal velocity and not an outlet: a wall at rest or moving."""
        if condition.bc_type == WALL:
            return True
        return (
            condition.bc_type == VELOCITY_INLET
            and StaggeredBoundary._normal_component(condition, edge) == 0.0
        )

    def _build_walls(self, edge: str) -> EdgeWalls:
        """The wall faces along an edge, per edge cell and per corner."""
        solid = self._solid_along(edge)
        cells = np.array(
            [self._is_wall(c, edge) for c in self._cell_conditions[edge]], dtype=bool
        )
        cells &= ~solid
        corners = np.array(
            [self._is_wall(c, edge) for c in self._corner_conditions(edge)],
            dtype=bool,
        )
        cells.flags.writeable = False
        corners.flags.writeable = False
        return EdgeWalls(edge=edge, cells=cells, corners=corners)

    def _build_outlet(self, edge: str) -> PressureOutletCondition:
        """Outlet mask along an edge; the value is the gauge datum."""
        is_outlet = np.array(
            [c.bc_type == PRESSURE_OUTLET for c in self._cell_conditions[edge]],
            dtype=bool,
        )
        is_outlet.flags.writeable = False
        return PressureOutletCondition(edge=edge, is_outlet=is_outlet, pressure=0.0)

    # ------------------------------------------------------------------
    # Deliverable 1: exact imposition of the normal components
    # ------------------------------------------------------------------

    def apply_normal_velocity(self, u: np.ndarray, v: np.ndarray) -> None:
        """Write the Dirichlet normal velocities into the domain-face entries.

        Wall faces get zero and inlet faces get the prescribed component,
        exactly. Pressure outlet faces are not written: their normal
        velocity is not a Dirichlet condition and is left to the momentum
        and pressure steps, which read ``pressure_outlets``. The faces of
        every vent (ADR-014 J), interior entries on obstacle faces, get the
        vent's normal velocity. No other interior entry and no tangential
        entry is touched.

        Parameters
        ----------
        u : np.ndarray
            x-velocity on vertical faces, shape [ny, nx+1], modified in-place
            at columns 0 and nx and at the side vents' faces only.
        v : np.ndarray
            y-velocity on horizontal faces, shape [ny+1, nx], modified
            in-place at rows 0 and ny and at the top and bottom vents' faces
            only.

        Raises
        ------
        ValueError
            If either array does not have the staggered shape for this mesh.
        """
        if u.shape != self._u_shape or v.shape != self._v_shape:
            raise ValueError(
                f"expected staggered shapes u {self._u_shape} and v {self._v_shape}, "
                f"got u {u.shape} and v {v.shape}"
            )
        mask = self._normal_dirichlet["bottom"]
        v[0, mask] = self._normal_value["bottom"][mask]
        mask = self._normal_dirichlet["top"]
        v[-1, mask] = self._normal_value["top"][mask]
        mask = self._normal_dirichlet["left"]
        u[mask, 0] = self._normal_value["left"][mask]
        mask = self._normal_dirichlet["right"]
        u[mask, -1] = self._normal_value["right"][mask]
        for run in self._vents:
            target = u if run.component == "u" else v
            target[run.rows, run.cols] = run.normal

    # ------------------------------------------------------------------
    # Deliverable 2: the remaining conditions as data
    # ------------------------------------------------------------------

    def tangential_conditions(self) -> dict[str, TangentialCondition]:
        """Tangential Dirichlet data for every edge, keyed by edge name.

        Returns
        -------
        dict[str, TangentialCondition]
            Keys "bottom", "top", "left", "right". The arrays are read-only.
        """
        return dict(self._tangential)

    def pressure_outlets(self) -> dict[str, PressureOutletCondition]:
        """Pressure outlet data for every edge, keyed by edge name.

        Returns
        -------
        dict[str, PressureOutletCondition]
            Keys "bottom", "top", "left", "right". The arrays are read-only.
        """
        return dict(self._outlets)

    def wall_faces(self) -> dict[str, EdgeWalls]:
        """Which edge faces are walls, keyed by edge name (ECR-002 step 6).

        Returns
        -------
        dict[str, EdgeWalls]
            Keys "bottom", "top", "left", "right". The arrays are read-only.
        """
        return dict(self._walls)

    def fixed_flow_velocities(self) -> dict[str, float]:
        """Outward normal velocity of every fixed-flow outlet segment, by name.

        Returns
        -------
        dict[str, float]
            Segment name to velocity in m/s, positive out of the domain, in
            configuration order: the stated value, or the equal share of
            the remaining discrete inflow for a segment that states none.
            Empty when the configuration has no fixed-flow outlet. A copy.
        """
        return dict(self._fixed_flow)

    def vent_runs(self) -> list[VentRun]:
        """The faces of every vent segment, in configuration order (ADR-014 J).

        Returns
        -------
        list[VentRun]
            ``vent_runs(mesh, config, registry)`` as built once here; empty
            without vents. A copy of the list.
        """
        return list(self._vents)

    def vent_faces(self) -> VentFaces:
        """Which faces are vents, normal and tangential (ADR-014 J).

        Returns
        -------
        VentFaces
            Read-only masks, all False without vents.
        """
        return self._vent_faces

    def has_pressure_outlet(self) -> bool:
        """Return True if any edge cell is a pressure outlet.

        Returns
        -------
        bool
            True when at least one edge face is an outlet; a closed domain
            (lid-driven cavity) returns False.
        """
        return any(bool(np.any(o.is_outlet)) for o in self._outlets.values())

    # ------------------------------------------------------------------
    # Fluxes and scales
    # ------------------------------------------------------------------

    def get_inlet_flux(self, boundary_name: str) -> float:
        """Volumetric flux into the domain through a named velocity inlet.

        The sum over the faces the shared coverage gives to this segment of
        the prescribed normal velocity, signed into the domain, times the
        face width from the mesh. Every term is a face value at the domain
        boundary, so the sum is exact on uniform and stretched meshes alike,
        and every face is counted for one segment at most.

        Parameters
        ----------
        boundary_name : str
            Name of a boundary defined in the config (e.g., "hepa_supply").

        Returns
        -------
        float
            Volumetric flux in m^2/s (per unit depth), positive into the
            domain. Zero for a segment that is not a velocity inlet, and
            zero for a purely tangential inlet such as a moving lid. A
            discharge vent's is its velocity times its faces' widths
            (ADR-014 J).

        Raises
        ------
        KeyError
            If boundary_name is not found in the config.
        """
        spec = self._registry.spec(boundary_name)
        if spec.location == OBSTACLE:
            return self._vent_inlet_flux(boundary_name)
        edge = spec.location
        flux = 0.0
        widths = self._face_widths(edge)
        for k, point in enumerate(self._cell_coverage[edge]):
            if point.name != boundary_name:
                continue
            if point.condition.bc_type != VELOCITY_INLET:
                continue
            flux += (
                _INWARD_SIGN[edge]
                * self._normal_component(point.condition, edge)
                * float(widths[k])
            )
        return flux

    def _vent_inlet_flux(self, name: str) -> float:
        """Inflow through a discharge vent's faces; zero for an exhaust (ADR-014 J)."""
        flux = 0.0
        for run in self._vents:
            if run.name != name or not run.is_inlet:
                continue
            for width in run.widths:
                flux += _INWARD_SIGN[run.edge] * run.normal * float(width)
        return flux

    def get_total_inlet_flux(self) -> float:
        """Volumetric flux into the domain summed over every velocity inlet.

        Returns
        -------
        float
            Sum of get_inlet_flux(name) over all boundaries of type
            velocity_inlet, discharge vents included, in m^2/s (per unit
            depth).
        """
        total = 0.0
        for name, spec in self._registry.boundaries.items():
            if spec.type == VELOCITY_INLET:
                total += self.get_inlet_flux(name)
        return total

    def get_max_boundary_velocity(self) -> float:
        """Largest velocity the room is driven by: inlets and moving walls.

        Both components count, so a tangential lid sets the scale of a
        closed cavity. SOLID edge cells contribute zero. An outlet's
        velocity is what leaves, so a fixed-flow outlet does not set the
        scale, whether it states its velocity or shares the remainder. A
        configuration with a fixed-flow outlet must have a velocity inlet
        (config.py), so the scale is never left to the outlets.

        Returns
        -------
        float
            Maximum of |u_prescribed| and |v_prescribed| over the edge
            cells that are not fixed-flow outlets and over the discharge
            vents, inlets too (ADR-014 J); 0.0 when nothing is prescribed.
        """
        max_vel = 0.0
        for conditions in self._cell_conditions.values():
            for c in conditions:
                if c.bc_type == FIXED_FLOW_OUTLET:
                    continue
                max_vel = max(max_vel, abs(c.u_prescribed), abs(c.v_prescribed))
        for run in self._vents:
            if run.is_inlet:
                max_vel = max(max_vel, abs(run.normal))
        return max_vel
