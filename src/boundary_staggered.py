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
puts there, because the obstacle is what bounds the flow at that face.
"""

from dataclasses import dataclass

import numpy as np

from src.boundary_registry import (
    EDGES,
    NO_SLIP_WALL,
    PRESSURE_OUTLET,
    VELOCITY_INLET,
    BoundaryRegistry,
    EdgeCondition,
    covers,
)
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.staggered import u_shape, v_shape

# Sign that turns the prescribed normal component into flow into the domain.
_INWARD_SIGN: dict[str, float] = {
    "bottom": 1.0,
    "top": -1.0,
    "left": 1.0,
    "right": -1.0,
}


@dataclass(frozen=True)
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
        prescribed (wall, velocity_inlet, or a face of a SOLID edge cell);
        False at pressure outlets, where the component has zero gradient.
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
    """

    def __init__(self, mesh: Mesh, config: SimConfig) -> None:
        self._mesh = mesh
        self._registry = BoundaryRegistry(config)
        self._u_shape = u_shape(mesh)
        self._v_shape = v_shape(mesh)

        # Conditions at the cell positions along each edge, the SOLID rule
        # applied once here so every query below sees the same answer.
        self._cell_conditions: dict[str, list[EdgeCondition]] = {}
        for edge in EDGES:
            solid = self._solid_along(edge)
            self._cell_conditions[edge] = [
                NO_SLIP_WALL if is_solid else self._registry.condition_at(edge, c)
                for c, is_solid in zip(self._cell_coordinates(edge), solid, strict=True)
            ]

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

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _cell_coordinates(self, edge: str) -> np.ndarray:
        """Cell-center coordinates along an edge: xc on top/bottom, yc on left/right."""
        if edge in ("top", "bottom"):
            return self._mesh.xc
        return self._mesh.yc

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
        ct = self._mesh.cell_type
        if edge == "bottom":
            return ct[0, :] == SOLID
        if edge == "top":
            return ct[-1, :] == SOLID
        if edge == "left":
            return ct[:, 0] == SOLID
        return ct[:, -1] == SOLID

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

    # ------------------------------------------------------------------
    # Construction of the data deliverables
    # ------------------------------------------------------------------

    def _build_tangential(self, edge: str) -> TangentialCondition:
        """Tangential data at every storage location along an edge.

        The condition at a storage coordinate is the registry's answer at
        that coordinate, so at a point where two segments meet the first in
        configuration order wins, the same rule as everywhere else. A storage
        location that bounds a SOLID edge cell is a wall.
        """
        solid = self._solid_along(edge)
        solid_face = np.zeros(solid.shape[0] + 1, dtype=bool)
        solid_face[:-1] |= solid
        solid_face[1:] |= solid

        conditions = [
            NO_SLIP_WALL if is_solid else self._registry.condition_at(edge, c)
            for c, is_solid in zip(
                self._face_coordinates(edge), solid_face, strict=True
            )
        ]
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
        and pressure steps, which read ``pressure_outlets``. No interior
        entry and no tangential entry is touched.

        Parameters
        ----------
        u : np.ndarray
            x-velocity on vertical faces, shape [ny, nx+1], modified in-place
            at columns 0 and nx only.
        v : np.ndarray
            y-velocity on horizontal faces, shape [ny+1, nx], modified
            in-place at rows 0 and ny only.

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

        The sum over the inlet's edge faces of the prescribed normal
        velocity, signed into the domain, times the face width from the
        mesh. Every term is a face value at the domain boundary, so the sum
        is exact on uniform and stretched meshes alike.

        Parameters
        ----------
        boundary_name : str
            Name of a boundary defined in the config (e.g., "hepa_supply").

        Returns
        -------
        float
            Volumetric flux in m^2/s (per unit depth), positive into the
            domain. Zero for a segment that is not a velocity inlet, and
            zero for a purely tangential inlet such as a moving lid.

        Raises
        ------
        KeyError
            If boundary_name is not found in the config.
        """
        spec = self._registry.spec(boundary_name)
        edge = spec.location
        flux = 0.0
        widths = self._face_widths(edge)
        for k, (coord, condition) in enumerate(
            zip(self._cell_coordinates(edge), self._cell_conditions[edge], strict=True)
        ):
            if condition.bc_type != VELOCITY_INLET:
                continue
            if not covers(spec, edge, float(coord)):
                continue
            flux += (
                _INWARD_SIGN[edge]
                * self._normal_component(condition, edge)
                * float(widths[k])
            )
        return flux

    def get_total_inlet_flux(self) -> float:
        """Volumetric flux into the domain summed over every velocity inlet.

        Returns
        -------
        float
            Sum of get_inlet_flux(name) over all boundaries of type
            velocity_inlet, in m^2/s (per unit depth).
        """
        total = 0.0
        for name, spec in self._registry.boundaries.items():
            if spec.type == VELOCITY_INLET:
                total += self.get_inlet_flux(name)
        return total

    def get_max_boundary_velocity(self) -> float:
        """Largest absolute prescribed velocity component on any edge.

        Both components count, so a tangential lid sets the scale of a
        closed cavity. SOLID edge cells contribute zero.

        Returns
        -------
        float
            Maximum of |u_prescribed| and |v_prescribed| over all edge
            cells; 0.0 when nothing is prescribed.
        """
        max_vel = 0.0
        for conditions in self._cell_conditions.values():
            for c in conditions:
                max_vel = max(max_vel, abs(c.u_prescribed), abs(c.v_prescribed))
        return max_vel
