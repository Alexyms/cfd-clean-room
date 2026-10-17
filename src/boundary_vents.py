"""The mesh faces of every equipment vent, the one derivation each layer reads (ADR-014 J).

A vent is a boundary segment on an obstacle face (``location: obstacle``).
Its faces are interior entries of the face arrays, the faces between the
obstacle's cells and the room that ``staggered.obstacle_face_inputs`` finds,
and which vent covers each is the registry's ``coverage_on``, the same
first-match rule the domain edges follow. ``vent_runs`` combines the two and
``vent_face_masks`` turns the runs into the masks the momentum, turbulence
and concentration layers treat apart. Nothing here reads a field or the
velocity layer's state, so the velocity layer (src/boundary_staggered.py)
and the concentration layer (src/boundary_concentration.py) both import
this module and neither imports the other for it (REQ-S12.1; review 61 S2,
moved here at ADR-014 step 2).
"""

from dataclasses import dataclass

import numpy as np

from src.boundary_registry import (
    ROOM_EDGE_OF_FACE,
    VELOCITY_INLET,
    BoundaryRegistry,
    EdgeCondition,
)
from src.config import MESH_BALANCE_TOLERANCE, BoundarySpec, SimConfig
from src.mesh import Mesh
from src.staggered import ObstacleFace, obstacle_face_inputs, u_shape, v_shape


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
        (``boundary_registry.ROOM_EDGE_OF_FACE``): "bottom" for a top face.
    component : str
        "v" on a top or bottom face, "u" on a side.
    rows, cols : np.ndarray
        int64, the vent's faces in its component's array, in order along
        the face. Read-only, as every array here: the velocity, turbulence
        and concentration layers share one run.
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
        rule that needs the mesh), or a recirculating pair's flows differ on
        the faces they cover.
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
        arrays = {
            "rows": inputs.rows[mine],
            "cols": inputs.cols[mine],
            "widths": inputs.widths[mine],
            "corner_rows": inputs.corner_rows[mine_corner],
            "corner_cols": inputs.corner_cols[mine_corner],
        }
        for array in arrays.values():
            array.flags.writeable = False
        runs.append(
            VentRun(
                name=name,
                spec=spec,
                edge=ROOM_EDGE_OF_FACE[spec.face],
                component=inputs.component,
                condition=condition,
                **arrays,
            )
        )
    _check_recirculated_flows(runs)
    return runs


def _check_recirculated_flows(runs: list[VentRun]) -> None:
    """Raise unless each recirculating pair's flows agree on the mesh's faces.

    Load compared the configured extents; the faces may round the two
    openings differently, and a discharge of more flow than its intake
    creates particles at the intake's concentration (review 61 S6).
    """
    by_name = {run.name: run for run in runs}
    for run in runs:
        source = run.spec.recirculate_from
        if not run.is_inlet or source is None:
            continue
        intake = by_name[source]
        returned = abs(run.normal) * float(run.widths.sum())
        drawn = abs(intake.normal) * float(intake.widths.sum())
        if abs(returned - drawn) > MESH_BALANCE_TOLERANCE * max(returned, drawn):
            raise ValueError(
                f"boundaries.{run.name}: on this mesh's faces the discharge's flow "
                f"{returned:.15g} m^2/s differs from its intake '{source}''s "
                f"{drawn:.15g} m^2/s; place the vents' ends on cell faces"
            )


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
