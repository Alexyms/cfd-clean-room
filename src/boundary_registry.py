"""Configuration interpretation shared by the collocated and staggered boundary layers.

Which condition holds at a point on a domain edge is a question about the
configuration and the geometry: which named boundary segment covers that
point, what type it is, and what velocity it prescribes there. None of it
depends on where a solver stores its unknowns. This module answers that
question once, so the collocated ghost-cell layer in src/boundary.py and
the staggered imposition layer in src/boundary_staggered.py read the same
interpretation and cannot drift apart while both exist (REQ-S12.1).

The rules are the ones the collocated layer has always applied. A segment
covers a point when it sits on the same edge and the point's coordinate
along that edge lies within [start, end], inclusive at both ends. The first
covering segment in configuration order wins. A point no segment covers is
a no-slip wall. Nothing here reads a mesh or a field.
"""

from dataclasses import dataclass

from src.config import BoundarySpec, SimConfig

EDGES: tuple[str, str, str, str] = ("bottom", "top", "left", "right")

WALL: str = "wall"
VELOCITY_INLET: str = "velocity_inlet"
PRESSURE_OUTLET: str = "pressure_outlet"


@dataclass(frozen=True)
class EdgeCondition:
    """The condition at one point of a domain edge.

    Parameters
    ----------
    bc_type : str
        One of "wall", "velocity_inlet", "pressure_outlet".
    u_prescribed : float
        x-velocity at the domain face. Zero for wall and pressure_outlet.
    v_prescribed : float
        y-velocity at the domain face. Zero for wall and pressure_outlet.
    """

    bc_type: str
    u_prescribed: float
    v_prescribed: float


NO_SLIP_WALL: EdgeCondition = EdgeCondition(WALL, 0.0, 0.0)


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
        If the segment type is not one of the three known types.

    Notes
    -----
    A velocity inlet with explicit ``u_velocity`` or ``v_velocity`` uses
    them directly, with a missing component read as zero; this is how a
    tangential lid is written. Otherwise the ``velocity`` magnitude is
    decomposed normal to the edge, pointing into the domain.
    """
    if spec.type == WALL:
        return NO_SLIP_WALL

    if spec.type == PRESSURE_OUTLET:
        return EdgeCondition(PRESSURE_OUTLET, 0.0, 0.0)

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
        for spec in self._boundaries.values():
            if covers(spec, edge, coordinate):
                return condition_of(spec, edge)
        return NO_SLIP_WALL
