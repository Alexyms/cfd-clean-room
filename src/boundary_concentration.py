"""Concentration conditions on the staggered faces, as data (ADR-011 E).

The registry's second reader (REQ-S12.1). The velocity layer in
src/boundary_staggered.py decides which domain faces are walls, inlets and
outlets by asking ``BoundaryRegistry.coverage_along`` on the coordinates
and SOLID mask ``staggered.edge_cell_inputs`` derives; this module makes
the same call, and derives from the answer the scalar condition the
transport solver applies at every face. It never reads or writes a
concentration field, and it imposes nothing: it hands the solver arrays
shaped like the staggered faces and the solver forms the fluxes.

The conditions, per face, for particle class k:

- Inlet (a ``velocity_inlet`` segment with a nonzero normal velocity): the
  inward flux carries ``concentration[k]``, times one minus
  ``hepa_efficiency(k)`` when the segment is ``hepa_filtered``; zero when
  the key is absent, which is the clean supply of ADR-011 decision 3.
  ``inflow_u``, ``inflow_v``. A ``velocity_inlet`` whose normal velocity
  is zero (a tangential lid) admits no air and is a moving surface
  particles land on: a wall below (ADR-011 E, amended 2026-10-03).
- Outlet (``pressure_outlet``): nothing is carried in; a reversed outlet
  face brings clean air. The outflow is the solver's, from the upwind cell.
- Wall: a deposition flux at ``deposition_velocity(k, surface)`` where the
  surface is the edge's own (bottom is floor, top is ceiling, left and right
  are walls) unless the covering wall segment names a ``deposition_surface``;
  ``none`` switches the flux off. ``deposition_u``, ``deposition_v``.
- Faces between a non-SOLID cell and a SOLID cell: walls of the
  orientation the face normal gives, the top of an obstacle a floor, its
  underside a ceiling, its sides walls, each with the deposition velocity
  for that surface as its single removal (ADR-011 D). A face of a SOLID
  cell on the domain edge, and a face between two SOLID cells, is nothing.
- ``surface_u``, ``surface_v`` name the surface each depositing face is
  booked to by the codes below, ``SURFACE_NONE`` elsewhere. Obstacle faces
  carry the orientation code; they are interior faces (index 1 to n-1 along
  the normal axis) while domain faces are index 0 and n, which is how the
  solver books obstacle deposition apart.
- ``settling_v`` marks the horizontal faces that carry the class's settling
  increment: both neighbouring cells non-SOLID (the BOUNDARY ring counts),
  so never the domain's top or bottom faces and never a face that meets a
  SOLID cell. There is no ``settling_u``: settling acts in -y.

The surface codes, the edge defaults and the dataclass are this module's;
the surface names come from src/config.py, which validates the segment key.
"""

from dataclasses import dataclass

import numpy as np

from src.boundary_registry import (
    EDGES,
    VELOCITY_INLET,
    WALL,
    BoundaryRegistry,
    EdgeCoverage,
)
from src.config import (
    DEPOSITION_CEILING,
    DEPOSITION_FLOOR,
    DEPOSITION_NONE,
    DEPOSITION_WALL,
    BoundarySpec,
    SimConfig,
)
from src.mesh import SOLID, Mesh
from src.particles import ParticlePhysics
from src.staggered import edge_cell_inputs, u_shape, v_shape

# The small integers surface_u and surface_v carry (ADR-011 E).
SURFACE_NONE: int = 0
SURFACE_FLOOR: int = 1
SURFACE_CEILING: int = 2
SURFACE_WALL: int = 3
SURFACE_CODES: dict[str, int] = {
    DEPOSITION_NONE: SURFACE_NONE,
    DEPOSITION_FLOOR: SURFACE_FLOOR,
    DEPOSITION_CEILING: SURFACE_CEILING,
    DEPOSITION_WALL: SURFACE_WALL,
}

# The surface a domain edge deposits to unless a wall segment overrides it.
EDGE_SURFACES: dict[str, str] = {
    "bottom": DEPOSITION_FLOOR,
    "top": DEPOSITION_CEILING,
    "left": DEPOSITION_WALL,
    "right": DEPOSITION_WALL,
}


@dataclass(frozen=True, eq=False)
class ConcentrationFaces:
    """The scalar conditions of one particle class on every staggered face.

    Every array is read-only and shaped like the face it describes:
    ``[ny, nx+1]`` for the ``_u`` arrays (vertical faces) and ``[ny+1, nx]``
    for the ``_v`` arrays (horizontal faces).

    Parameters
    ----------
    inflow_u, inflow_v : np.ndarray
        float64. The concentration an inward flux carries, particles per
        cubic meter; nonzero only at velocity_inlet faces with a configured
        concentration.
    deposition_u, deposition_v : np.ndarray
        float64. The deposition velocity in m/s at every wall face, domain
        and SOLID-adjacent; zero elsewhere and on a ``none`` surface.
    surface_u, surface_v : np.ndarray
        int32. The surface code each depositing face is booked to:
        SURFACE_FLOOR, SURFACE_CEILING or SURFACE_WALL; SURFACE_NONE
        elsewhere.
    settling_v : np.ndarray
        bool. True on the horizontal faces that carry the class's settling
        increment.
    """

    inflow_u: np.ndarray
    inflow_v: np.ndarray
    deposition_u: np.ndarray
    deposition_v: np.ndarray
    surface_u: np.ndarray
    surface_v: np.ndarray
    settling_v: np.ndarray


@dataclass(frozen=True)
class _InletRun:
    """The faces one inlet segment covers on one component, by index."""

    component: str
    rows: np.ndarray
    cols: np.ndarray
    spec: BoundarySpec


def _freeze(array: np.ndarray) -> np.ndarray:
    """Clear the writeable flag and return the array."""
    array.flags.writeable = False
    return array


class ConcentrationBoundary:
    """Derive the per-face concentration conditions for every particle class.

    Parameters
    ----------
    mesh : Mesh
        Supplies the cell coordinates along each edge, the face shapes and
        ``cell_type``.
    config : SimConfig
        Supplies the number of particle classes.
    physics : ParticlePhysics
        Supplies ``deposition_velocity`` and ``hepa_efficiency`` per class.
    registry : BoundaryRegistry
        The one interpretation of the configured segments, shared with the
        velocity layer.

    Raises
    ------
    ValueError
        If ``physics`` was built for a different number of classes than
        ``config`` names.
    """

    def __init__(
        self,
        mesh: Mesh,
        config: SimConfig,
        physics: ParticlePhysics,
        registry: BoundaryRegistry,
    ) -> None:
        self._physics = physics
        self._n_classes = len(config.particle_sizes)
        if physics.n_classes != self._n_classes:
            raise ValueError(
                f"physics was built for {physics.n_classes} classes but the "
                f"configuration names {self._n_classes}"
            )
        self._u_shape = u_shape(mesh)
        self._v_shape = v_shape(mesh)
        ny, nx = mesh.cell_type.shape

        surface_u = np.zeros(self._u_shape, dtype=np.int32)
        surface_v = np.zeros(self._v_shape, dtype=np.int32)
        self._inlets: list[_InletRun] = []
        for edge in EDGES:
            coverage = registry.coverage_along(edge, *edge_cell_inputs(mesh, edge))
            self._classify_edge(edge, coverage, surface_u, surface_v, ny, nx)

        # Faces between a non-SOLID and a SOLID cell: walls of the orientation
        # the normal gives. The slices are views, so the assignments land in
        # the full arrays at the interior face indices 1 to n-1.
        solid = mesh.cell_type == SOLID
        west, east = solid[:, :-1], solid[:, 1:]
        surface_u[:, 1:-1][west ^ east] = SURFACE_WALL
        south, north = solid[:-1, :], solid[1:, :]
        surface_v[1:-1, :][south & ~north] = SURFACE_FLOOR
        surface_v[1:-1, :][~south & north] = SURFACE_CEILING

        settling_v = np.zeros(self._v_shape, dtype=bool)
        settling_v[1:-1, :] = ~south & ~north

        self._surface_u = _freeze(surface_u)
        self._surface_v = _freeze(surface_v)
        self._settling_v = _freeze(settling_v)

    def _classify_edge(
        self,
        edge: str,
        coverage: list[EdgeCoverage],
        surface_u: np.ndarray,
        surface_v: np.ndarray,
        ny: int,
        nx: int,
    ) -> None:
        """Book the wall faces of one edge and record its inlet runs.

        A SOLID edge cell has no scalar face. An outlet face carries
        nothing in. A wall face takes the segment's ``deposition_surface``
        when one is named, else the edge's default. An inlet whose normal
        velocity is zero is a wall here, with the edge's default surface.
        """
        inlet_positions: dict[str, list[int]] = {}
        inlet_specs: dict[str, BoundarySpec] = {}
        for k, point in enumerate(coverage):
            if point.solid:
                continue
            bc_type = point.condition.bc_type
            if bc_type == VELOCITY_INLET and self._normal(point, edge) != 0.0:
                assert point.name is not None and point.spec is not None
                inlet_positions.setdefault(point.name, []).append(k)
                inlet_specs[point.name] = point.spec
            elif bc_type in (WALL, VELOCITY_INLET):
                surface = EDGE_SURFACES[edge]
                if point.spec is not None and point.spec.deposition_surface is not None:
                    surface = point.spec.deposition_surface
                component, rows, cols = self._edge_face_index(
                    edge, np.array([k]), ny, nx
                )
                target = surface_u if component == "u" else surface_v
                target[rows, cols] = SURFACE_CODES[surface]
        for name, positions in inlet_positions.items():
            component, rows, cols = self._edge_face_index(
                edge, np.array(positions), ny, nx
            )
            self._inlets.append(_InletRun(component, rows, cols, inlet_specs[name]))

    @staticmethod
    def _normal(point: EdgeCoverage, edge: str) -> float:
        """The prescribed component normal to an edge: v on top/bottom, u on left/right."""
        if edge in ("bottom", "top"):
            return point.condition.v_prescribed
        return point.condition.u_prescribed

    @staticmethod
    def _edge_face_index(
        edge: str, positions: np.ndarray, ny: int, nx: int
    ) -> tuple[str, np.ndarray, np.ndarray]:
        """The component and (rows, cols) of the edge faces at these positions."""
        fixed = np.full_like(
            positions, {"bottom": 0, "top": ny, "left": 0, "right": nx}[edge]
        )
        if edge in ("bottom", "top"):
            return "v", fixed, positions
        return "u", positions, fixed

    def _carried(self, spec: BoundarySpec, size_class: int) -> float:
        """The concentration an inlet segment carries for one class."""
        if spec.concentration is None:
            return 0.0
        value = spec.concentration[size_class]
        if spec.hepa_filtered:
            value *= 1.0 - self._physics.hepa_efficiency(size_class)
        return value

    def faces_for(self, size_class: int) -> ConcentrationFaces:
        """The scalar conditions of one class on every face.

        Parameters
        ----------
        size_class : int
            Index into the configured particle sizes.

        Returns
        -------
        ConcentrationFaces
            Fresh read-only inflow and deposition arrays for the class; the
            surface codes and the settling mask are the same read-only
            arrays for every class, since neither depends on it.

        Raises
        ------
        TypeError
            If ``size_class`` is not an int (a bool is not one here).
        IndexError
            If ``size_class`` is outside the configured range.
        """
        if isinstance(size_class, bool) or not isinstance(size_class, int):
            raise TypeError(
                f"size_class must be an int, got {type(size_class).__name__}"
            )
        if not 0 <= size_class < self._n_classes:
            raise IndexError(
                f"size_class {size_class} out of range [0, {self._n_classes - 1}]"
            )
        velocity = np.zeros(len(SURFACE_CODES), dtype=np.float64)
        for name, code in SURFACE_CODES.items():
            if name != DEPOSITION_NONE:
                velocity[code] = self._physics.deposition_velocity(size_class, name)
        inflow_u = np.zeros(self._u_shape, dtype=np.float64)
        inflow_v = np.zeros(self._v_shape, dtype=np.float64)
        for run in self._inlets:
            target = inflow_u if run.component == "u" else inflow_v
            target[run.rows, run.cols] = self._carried(run.spec, size_class)
        return ConcentrationFaces(
            inflow_u=_freeze(inflow_u),
            inflow_v=_freeze(inflow_v),
            deposition_u=_freeze(np.ascontiguousarray(velocity[self._surface_u])),
            deposition_v=_freeze(np.ascontiguousarray(velocity[self._surface_v])),
            surface_u=self._surface_u,
            surface_v=self._surface_v,
            settling_v=self._settling_v,
        )
