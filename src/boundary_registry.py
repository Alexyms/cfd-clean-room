"""Configuration interpretation shared by every boundary imposition layer.

Which condition holds at a point on a domain edge is a question about the
configuration and the geometry: which named boundary segment covers that
point, what type it is, and what velocity it prescribes there. None of it
depends on where a solver stores its unknowns. This module answers that
question once, so the staggered velocity layer in src/boundary_staggered.py
and the concentration layer in src/boundary_concentration.py read the same
interpretation and cannot drift apart (REQ-S12.1).

The rules are the ones the project's first boundary layer applied. A segment
covers a point when it sits on the same edge and the point's coordinate
along that edge lies within [start, end], inclusive at both ends. The first
covering segment in configuration order wins. A point no segment covers is
a no-slip wall, and so is a point behind which the cell is SOLID, because
the obstacle is what bounds the flow at that face whatever segment the
configuration puts there. ``coverage_along`` applies all of that to a run of
points at once and is the one derivation of which faces a segment covers:
the velocity layer builds its per-cell and per-corner conditions from it
and the concentration layer its per-face conditions, so an inlet face in
one layer is an inlet face in the other. Nothing here reads a mesh or a
field: a layer reads the coordinates along an edge and the SOLID cells
behind them from the mesh and hands both in.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.config import BoundarySpec, SimConfig

EDGES: tuple[str, str, str, str] = ("bottom", "top", "left", "right")

WALL: str = "wall"
VELOCITY_INLET: str = "velocity_inlet"
PRESSURE_OUTLET: str = "pressure_outlet"
FIXED_FLOW_OUTLET: str = "fixed_flow_outlet"


@dataclass(frozen=True)
class EdgeCondition:
    """The condition at one point of a domain edge.

    Parameters
    ----------
    bc_type : str
        One of "wall", "velocity_inlet", "pressure_outlet",
        "fixed_flow_outlet".
    u_prescribed : float
        x-velocity at the domain face. Zero for wall and pressure_outlet.
    v_prescribed : float
        y-velocity at the domain face. Zero for wall and pressure_outlet.

    Notes
    -----
    A fixed_flow_outlet carries its outward normal velocity here, but only
    a stated one: a segment that states none shares the inflow the stated
    outlets leave, which depends on the mesh, so ``condition_of`` carries
    zero for it and the velocity layer replaces that with
    ``fixed_flow_condition`` once the share is resolved.
    """

    bc_type: str
    u_prescribed: float
    v_prescribed: float


NO_SLIP_WALL: EdgeCondition = EdgeCondition(WALL, 0.0, 0.0)


@dataclass(frozen=True)
class EdgeCoverage:
    """Which segment covers one point of a domain edge, and the condition there.

    Parameters
    ----------
    name : str or None
        The first covering segment in configuration order; None when no
        segment covers the point or the cell behind it is SOLID.
    spec : BoundarySpec or None
        That segment; None likewise.
    condition : EdgeCondition
        The condition at the point: the segment's, or NO_SLIP_WALL when
        ``spec`` is None.
    solid : bool
        True when the cell behind the point is SOLID. The face then bounds
        the obstacle rather than the flow: a wall for the velocity layer,
        and no face at all for the scalar layer.
    """

    name: str | None
    spec: BoundarySpec | None
    condition: EdgeCondition
    solid: bool


UNCOVERED: EdgeCoverage = EdgeCoverage(None, None, NO_SLIP_WALL, False)
BEHIND_SOLID: EdgeCoverage = EdgeCoverage(None, None, NO_SLIP_WALL, True)


def covers(spec: BoundarySpec, edge: str, coordinate: float) -> bool:
    """Whether a boundary segment covers a point on a domain edge.

    Parameters
    ----------
    spec : BoundarySpec
        The named segment.
    edge : str
        One of "bottom", "top", "left", "right".
    coordinate : float
        Position along the edge: x for the bottom and top edges, y for
        the left and right edges.

    Returns
    -------
    bool
        True when the segment sits on ``edge`` and its range is defined
        and contains ``coordinate``, inclusive at both ends.
    """
    if spec.location != edge:
        return False
    if edge in ("top", "bottom"):
        return (
            spec.x_start is not None
            and spec.x_end is not None
            and spec.x_start <= coordinate <= spec.x_end
        )
    return (
        spec.y_start is not None
        and spec.y_end is not None
        and spec.y_start <= coordinate <= spec.y_end
    )


def fixed_flow_condition(edge: str, speed: float) -> EdgeCondition:
    """The condition of a fixed-flow outlet face leaving a domain edge at a given speed.

    Parameters
    ----------
    edge : str
        One of "bottom", "top", "left", "right".
    speed : float
        Outward normal velocity in m/s, positive out of the domain.

    Returns
    -------
    EdgeCondition
        Type "fixed_flow_outlet" with the normal component signed along the
        coordinate axis (negative on the bottom and left edges) and a zero
        tangential component.
    """
    if edge == "top":
        return EdgeCondition(FIXED_FLOW_OUTLET, 0.0, speed)
    if edge == "bottom":
        return EdgeCondition(FIXED_FLOW_OUTLET, 0.0, -speed)
    if edge == "left":
        return EdgeCondition(FIXED_FLOW_OUTLET, -speed, 0.0)
    return EdgeCondition(FIXED_FLOW_OUTLET, speed, 0.0)


def condition_of(spec: BoundarySpec, edge: str) -> EdgeCondition:
    """The condition a segment prescribes on the edge it sits on.

    Parameters
    ----------
    spec : BoundarySpec
        The matched segment.
    edge : str
        One of "bottom", "top", "left", "right".

    Returns
    -------
    EdgeCondition
        Type and face velocity components.

    Raises
    ------
    ValueError
        If the segment type is not one of the four known types.

    Notes
    -----
    A velocity inlet with explicit ``u_velocity`` or ``v_velocity`` uses
    them directly, with a missing component read as zero; this is how a
    tangential lid is written. Otherwise the ``velocity`` magnitude is
    decomposed normal to the edge, pointing into the domain.

    A fixed-flow outlet carries its stated ``velocity`` pointing out of the
    domain, or zero when it states none. The share of a segment that
    states none is a property of the mesh and is not known here.
    """
    if spec.type == WALL:
        return NO_SLIP_WALL

    if spec.type == PRESSURE_OUTLET:
        return EdgeCondition(PRESSURE_OUTLET, 0.0, 0.0)

    if spec.type == FIXED_FLOW_OUTLET:
        return fixed_flow_condition(
            edge, spec.velocity if spec.velocity is not None else 0.0
        )

    if spec.type != VELOCITY_INLET:
        raise ValueError(f"Unrecognized boundary type: {spec.type}")

    if spec.u_velocity is not None or spec.v_velocity is not None:
        u_face = spec.u_velocity if spec.u_velocity is not None else 0.0
        v_face = spec.v_velocity if spec.v_velocity is not None else 0.0
        return EdgeCondition(VELOCITY_INLET, u_face, v_face)

    vel = spec.velocity if spec.velocity is not None else 0.0
    if edge == "top":
        return EdgeCondition(VELOCITY_INLET, 0.0, -vel)
    if edge == "bottom":
        return EdgeCondition(VELOCITY_INLET, 0.0, vel)
    if edge == "left":
        return EdgeCondition(VELOCITY_INLET, vel, 0.0)
    return EdgeCondition(VELOCITY_INLET, -vel, 0.0)


class BoundaryRegistry:
    """Named boundary segments from the configuration, queried by edge and position.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration with boundary definitions.
    """

    def __init__(self, config: SimConfig) -> None:
        self._boundaries: dict[str, BoundarySpec] = config.boundaries

    @property
    def boundaries(self) -> dict[str, BoundarySpec]:
        """The named segments, in configuration order."""
        return self._boundaries

    def spec(self, name: str) -> BoundarySpec:
        """Return the segment with the given name.

        Parameters
        ----------
        name : str
            Name of a boundary defined in the config (e.g., "hepa_supply").

        Returns
        -------
        BoundarySpec
            The named segment.

        Raises
        ------
        KeyError
            If no segment has that name.
        """
        if name not in self._boundaries:
            raise KeyError(f"Boundary '{name}' not found in config")
        return self._boundaries[name]

    def segment_at(
        self, edge: str, coordinate: float
    ) -> tuple[str, BoundarySpec] | None:
        """The first segment covering a point on a domain edge, with its name.

        Parameters
        ----------
        edge : str
            One of "bottom", "top", "left", "right".
        coordinate : float
            Position along the edge: x for the bottom and top edges, y for
            the left and right edges.

        Returns
        -------
        tuple[str, BoundarySpec] or None
            The first covering segment in configuration order, or None when
            no segment covers the point.
        """
        for name, spec in self._boundaries.items():
            if covers(spec, edge, coordinate):
                return name, spec
        return None

    def condition_at(self, edge: str, coordinate: float) -> EdgeCondition:
        """The condition at a point on a domain edge.

        Parameters
        ----------
        edge : str
            One of "bottom", "top", "left", "right".
        coordinate : float
            Position along the edge: x for the bottom and top edges, y for
            the left and right edges.

        Returns
        -------
        EdgeCondition
            From the first covering segment in configuration order, or a
            no-slip wall when no segment covers the point.
        """
        found = self.segment_at(edge, coordinate)
        if found is None:
            return NO_SLIP_WALL
        return condition_of(found[1], edge)

    def coverage_along(
        self,
        edge: str,
        coordinates: np.ndarray | Sequence[float],
        solid: np.ndarray | Sequence[bool],
    ) -> list[EdgeCoverage]:
        """Coverage at each of a run of points along an edge, SOLID cells read as walls.

        The one derivation of which faces a segment covers (REQ-S12.1).
        Both boundary layers call it with the coordinates and SOLID mask
        ``staggered.edge_cell_inputs`` derives once, so they cannot disagree
        about a face.

        Parameters
        ----------
        edge : str
            One of "bottom", "top", "left", "right".
        coordinates : np.ndarray or Sequence[float]
            Positions along the edge in storage order: the cell centers for
            the edge's faces, or the face coordinates for its corners.
        solid : np.ndarray or Sequence[bool]
            For each point, whether the cell behind it is SOLID. Same
            length as ``coordinates``.

        Returns
        -------
        list[EdgeCoverage]
            One entry per point. A SOLID point is BEHIND_SOLID and an
            uncovered point is UNCOVERED, both no-slip walls with no
            segment; a covered point names its segment and carries
            ``condition_of`` that segment.

        Raises
        ------
        ValueError
            If the two sequences differ in length.
        """
        if len(coordinates) != len(solid):
            raise ValueError(
                f"coverage_along needs one SOLID flag per point, got "
                f"{len(coordinates)} points and {len(solid)} flags"
            )
        out: list[EdgeCoverage] = []
        for coordinate, is_solid in zip(coordinates, solid, strict=True):
            if is_solid:
                out.append(BEHIND_SOLID)
                continue
            found = self.segment_at(edge, float(coordinate))
            if found is None:
                out.append(UNCOVERED)
            else:
                name, spec = found
                out.append(EdgeCoverage(name, spec, condition_of(spec, edge), False))
        return out
