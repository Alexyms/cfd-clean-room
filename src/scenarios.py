"""The scenario manager: a scenario's intervals, flow states, sources and fields (ADR-014).

A scenario (``config.scenario``, ADR-014 B) is a merged base and a list of
events. Under tier 1 (ADR-014 A) the run is a sequence of intervals, each with
one steady flow state: the interval boundaries are every event's start and
end, so an event is active from its start and has expired at its end, and no
event starts or ends inside an interval. Each interval's configuration is the
merged base with its active replacements and sets applied.

Which intervals share a flow state is decided by a fingerprint of the keys the
velocity layer reads (``FLOW_KEYS``) over the interval's segments in
configuration order, since the coverage rule is first match in that order.
The keys only the transport reads (``TRANSPORT_KEYS``) are out, so a breach or
a dust release never adds a state, and a door that opens twice has two states,
not four solves. Contiguous segments (adjacent in configuration order, on one
edge or face, end to start, every other flow key equal) are read as one
segment over their union: a supply split into three panels for a breach of
the middle one keeps the base's fingerprint, as ADR-014 B's breach has one
flow state. The coverage rule gives the pieces the faces and the condition the
one segment would; what moves is the rounding of the inflow summed per
segment, which the returns' shared velocity carries (1.1e-15 relative on
160x60 and 4.0e-15 on 320x120 for the breach), so the joined state is the
split one to rounding, not bitwise. The number of distinct
states is the run's flow cost (T1's warm arm per state, ADR-014 decision 1)
and the manager reports it when it is built, before anything is solved.

Sources and the initial field are per class and per cell, on the mesh: a
source's rate is uniform over the non-SOLID cells whose centres lie in its
box, an emission is divided by those cells' area, and the initial field is
uniform over the non-SOLID cells. Nothing here solves, marches or writes:
that is time_integration's (ADR-014 G, step 4).
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from src.boundary_concentration import ConcentrationBoundary
from src.boundary_registry import BoundaryRegistry
from src.boundary_vents import vent_runs
from src.config import (
    BoundarySpec,
    EventSpec,
    Initial,
    ScenarioSpec,
    SimConfig,
    Source,
)
from src.mesh import SOLID, Mesh
from src.particles import ParticlePhysics

logger = logging.getLogger(__name__)

# The segment keys the velocity layer reads (prompt 62's decision 3): a state
# differs from another when one of these differs on a segment, or the
# segments' order does. The two inlet turbulence keys are here because the
# inflow k and eps change the flow under k-epsilon.
FLOW_KEYS: tuple[str, ...] = (
    "type",
    "location",
    "x_start",
    "x_end",
    "y_start",
    "y_end",
    "velocity",
    "u_velocity",
    "v_velocity",
    "obstacle",
    "face",
    "turbulence_intensity",
    "dissipation_length",
)
# The segment keys only the transport reads: an event that changes these
# alone is transport-only and keeps the flow state. deposition_surface is the
# wall's deposition, which the velocity layer does not read either.
TRANSPORT_KEYS: tuple[str, ...] = (
    "concentration",
    "hepa_filtered",
    "recirculate_from",
    "deposition_surface",
)
# The extent keys, which the contiguous-segment rule joins end to start.
_EXTENT_KEYS: frozenset[str] = frozenset({"x_start", "x_end", "y_start", "y_end"})


@dataclass(frozen=True, eq=False)
class Interval:
    """One interval of a scenario: a time span with one configuration.

    Parameters
    ----------
    t_start, t_end : float
        Seconds; the interval is [t_start, t_end).
    fingerprint : str
        The flow state's key (``flow_fingerprint``).
    mapping : dict
        The interval's merged mapping, a configuration file's structure. A
        copy: ``ScenarioManager.intervals`` builds new ones on each call.
    config : SimConfig
        ``mapping`` loaded.
    events : tuple[str, ...]
        The names of the events active over the interval, in file order.
    """

    t_start: float
    t_end: float
    fingerprint: str
    mapping: dict
    config: SimConfig
    events: tuple[str, ...]


def flow_segments(config: SimConfig) -> list[tuple[object, ...]]:
    """The segments as the velocity layer reads them, contiguous ones joined.

    Each segment becomes the tuple of its ``FLOW_KEYS`` values in that
    order. A segment that follows another in configuration order on the same
    edge or obstacle face, starts where it ends, and agrees with it on every
    flow key but the extent, is joined to it: the coverage rule gives the
    two the faces one segment over their union would, with one condition.

    Parameters
    ----------
    config : SimConfig
        A loaded configuration.

    Returns
    -------
    list[tuple[object, ...]]
        In configuration order.
    """
    out: list[dict[str, object]] = []
    for spec in config.boundaries.values():
        keys = {key: getattr(spec, key) for key in FLOW_KEYS}
        if out and _continues(out[-1], keys, spec):
            _, hi = _extent_names(spec)
            out[-1][hi] = keys[hi]
            continue
        out.append(keys)
    return [tuple(keys[key] for key in FLOW_KEYS) for keys in out]


def _extent_names(spec: BoundarySpec) -> tuple[str, str]:
    """The two extent keys a segment's range runs along."""
    if (spec.face or spec.location) in ("top", "bottom"):
        return "x_start", "x_end"
    return "y_start", "y_end"


def _continues(
    previous: dict[str, object], keys: dict[str, object], spec: BoundarySpec
) -> bool:
    """Whether a segment extends the previous one end to start with equal flow keys."""
    lo, hi = _extent_names(spec)
    same = all(previous[k] == keys[k] for k in FLOW_KEYS if k not in _EXTENT_KEYS)
    return same and previous[hi] == keys[lo] and previous[hi] is not None


def flow_fingerprint(config: SimConfig) -> str:
    """The key of a configuration's flow state (ADR-014 A).

    Parameters
    ----------
    config : SimConfig
        A loaded configuration.

    Returns
    -------
    str
        SHA-256, hexadecimal, of ``flow_segments`` in order. Equal for two
        configurations whose velocity layers read the same segments; the
        segment names, the transport-only keys and every section but the
        boundaries are out (within one scenario the other sections are the
        merged base's in every interval).
    """
    text = json.dumps(flow_segments(config), allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class ScenarioManager:
    """A scenario's intervals, flow states, sources and initial fields (ADR-014 G).

    Built before a run: it loads every interval's configuration, places the
    sources on the mesh and logs the number of distinct flow states.

    Parameters
    ----------
    config : SimConfig
        A configuration with a scenario section (``config.scenario``).
    mesh : Mesh
        The run's mesh. Every interval shares it: an event changes boundary
        segments, sources and fields, never the domain or the obstacles.

    Raises
    ------
    ValueError
        If ``config`` has no scenario, the mesh is not the configuration's
        grid, a source's box holds no non-SOLID cell centre on this mesh
        (ADR-014 B's rule that needs the mesh), or an interval's vent covers
        no face of this mesh or its recirculating pair's flows differ on the
        faces (ADR-014 J's).
    """

    def __init__(self, config: SimConfig, mesh: Mesh) -> None:
        if config.scenario is None:
            raise ValueError(
                "ScenarioManager needs a configuration with a scenario section "
                "(ADR-014 B)"
            )
        if mesh.cell_type.shape != (config.ny, config.nx):
            raise ValueError(
                f"the mesh is {mesh.cell_type.shape[1]}x{mesh.cell_type.shape[0]} but "
                f"the configuration's grid is {config.nx}x{config.ny}"
            )
        self._spec: ScenarioSpec = config.scenario
        self._mesh = mesh
        self._n_classes = len(config.particle_sizes)
        self._open = mesh.cell_type != SOLID
        self._volume = np.outer(mesh.dy_cell, mesh.dx_cell)

        times = self._spec.times()
        self._spans: list[tuple[float, float, tuple[str, ...]]] = []
        self._configs: dict[tuple[str, ...], tuple[SimConfig, str]] = {}
        for t0, t1 in pairwise(times):
            active = self._spec.active(t0)
            names = tuple(event.name for event in active)
            if names not in self._configs:
                interval_config = SimConfig.from_dict(self._spec.mapping(active))
                self._check_vents(interval_config, t0, t1, names)
                self._configs[names] = (
                    interval_config,
                    flow_fingerprint(interval_config),
                )
            self._spans.append((t0, t1, names))

        self._sources: list[tuple[EventSpec, np.ndarray]] = []
        for event in self._spec.events:
            if isinstance(event.change, Source):
                self._sources.append(
                    (event, self._source_rates(event.name, event.change))
                )
        self._initial: tuple[float, ...] = (0.0,) * self._n_classes
        for event in self._spec.events:
            if isinstance(event.change, Initial):
                self._initial = event.change.values

        logger.info(
            "scenario %s: %d intervals, %d distinct flow states",
            self._spec.path.name
            if self._spec.path is not None
            else f"on {self._spec.base_path.name}",
            len(self._spans),
            self.n_flow_states,
        )

    def _check_vents(
        self, config: SimConfig, t0: float, t1: float, names: tuple[str, ...]
    ) -> None:
        """Raise unless every vent of an interval covers a face of this mesh.

        ADR-014 J's rule that needs the mesh (``boundary_vents.vent_runs``):
        a vent's extent covers at least one face of its obstacle face, and a
        recirculating pair's flows agree on the faces. Checked here for every
        interval so a replacement vent that fails it is refused before a run,
        as the base's vents are when its boundary is built (review 62 S6).
        """
        try:
            vent_runs(self._mesh, config, BoundaryRegistry(config))
        except ValueError as err:
            raise ValueError(
                f"scenario: the configuration from {t0} s to {t1} s with "
                f"{list(names) if names else 'no events'} active: {err}"
            ) from err

    def _source_rates(self, name: str, source: Source) -> np.ndarray:
        """A source's rate per class and cell, [n_classes, ny, nx], read-only."""
        mesh = self._mesh
        in_x = (source.x_start <= mesh.xc) & (mesh.xc <= source.x_end)
        in_y = (source.y_start <= mesh.yc) & (mesh.yc <= source.y_end)
        cells = np.outer(in_y, in_x) & self._open
        if not cells.any():
            raise ValueError(
                f"scenario.events '{name}': the source's box [{source.x_start}, "
                f"{source.x_end}] x [{source.y_start}, {source.y_end}] holds no "
                "non-SOLID cell centre on this mesh"
            )
        if source.rate is not None:
            per_class = np.asarray(source.rate, dtype=np.float64)
        else:
            assert source.emission is not None
            area = float(self._volume[cells].sum())
            per_class = np.asarray(source.emission, dtype=np.float64) / area
        rates = np.zeros((self._n_classes, *cells.shape), dtype=np.float64)
        rates[:, cells] = per_class[:, None]
        rates.flags.writeable = False
        return rates

    @property
    def n_flow_states(self) -> int:
        """The number of distinct flow states, the run's flow cost in solves."""
        return len({fingerprint for _, fingerprint in self._configs.values()})

    def intervals(self) -> list[Interval]:
        """The scenario's intervals, in time order.

        Returns
        -------
        list[Interval]
            Covering [0, t_end) without gaps; consecutive intervals may share
            a fingerprint. New objects with copied mappings on every call.
        """
        out = []
        for t0, t1, names in self._spans:
            config, fingerprint = self._configs[names]
            out.append(
                Interval(
                    t_start=t0,
                    t_end=t1,
                    fingerprint=fingerprint,
                    mapping=self._spec.mapping(
                        tuple(e for e in self._spec.events if e.name in names)
                    ),
                    config=config,
                    events=names,
                )
            )
        return out

    def flow_states(self) -> dict[str, dict]:
        """Each distinct flow state's mapping, by fingerprint.

        Returns
        -------
        dict[str, dict]
            Fingerprint to the merged mapping of the first interval with it,
            in the order the states first appear. The flow solve reads its
            velocity-layer keys; its transport-only keys are that interval's.
        """
        out: dict[str, dict] = {}
        for interval in self.intervals():
            out.setdefault(interval.fingerprint, interval.mapping)
        return out

    def sources_at(self, t: float) -> dict[int, np.ndarray]:
        """The source rate of every class at time t, summed over active sources.

        A source is active from its start and has expired at its end, as
        every event is, to the time tolerance (``ScenarioSpec.is_active``).

        Parameters
        ----------
        t : float
            Seconds.

        Returns
        -------
        dict[int, np.ndarray]
            Class index to its rate, [ny, nx] float64, particles per cubic
            metre per second, zero in SOLID cells and where no source is
            active: ``solve_timestep``'s ``sources``. New arrays.
        """
        total = np.zeros((self._n_classes, *self._open.shape), dtype=np.float64)
        for event, rates in self._sources:
            if self._spec.is_active(event, t):
                total += rates
        return {k: np.ascontiguousarray(total[k]) for k in range(self._n_classes)}

    def conditions_for(self, interval: Interval) -> ConcentrationBoundary:
        """The concentration boundary of an interval's merged mapping.

        Parameters
        ----------
        interval : Interval
            One of ``intervals()``.

        Returns
        -------
        ConcentrationBoundary
            Built from the interval's configuration on the run's mesh: its
            inlets' concentrations, a breach's unfiltered panel included.
        """
        config = interval.config
        return ConcentrationBoundary(
            self._mesh, config, ParticlePhysics(config), BoundaryRegistry(config)
        )

    def initial_fields(self) -> dict[int, np.ndarray]:
        """The concentration of every class at t = 0.

        Returns
        -------
        dict[int, np.ndarray]
            Class index to [ny, nx] float64: the initial event's value per
            class over the non-SOLID cells, zero in SOLID cells; zero
            everywhere without an initial event. New arrays.
        """
        return {
            k: np.where(self._open, self._initial[k], 0.0)
            for k in range(self._n_classes)
        }
