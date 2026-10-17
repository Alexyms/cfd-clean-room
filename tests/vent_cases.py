"""Small rooms with equipment vents for the step 0b tests (ADR-014 J).

One room, 2 m by 1 m on 16 by 8 cells of 0.125 m, so every face position
can be read off by hand. A box stands on the floor over cells 6 to 9 in x
and 0 to 3 in y (x 0.75 to 1.25 m, y 0 to 0.5 m): its top face is the v
faces of row 4, columns 6 to 9; its left face the u faces of column 6, rows
0 to 3; its right face the u faces of column 10, rows 0 to 3. The supply
enters along the top and two returns on the floor share what the stated
outlets leave. The vents:

- ``DISCHARGE``: the box's top, x 0.875 to 1.125 (columns 7 and 8),
  0.5 m/s up, 0.125 m^2/s per metre.
- ``INTAKE``: the box's left side, y 0.125 to 0.375 (rows 1 and 2), 0.5 m/s
  into the box, the discharge's flow.
- ``EXHAUST``: the box's right side, y 0.125 to 0.375 (rows 1 and 2),
  0.4 m/s into the box, 0.1 m^2/s per metre.

A second room for the other orientations (prompt 61b, review 61 B1): the same
room with a shelf standing clear of the floor right of the box, over cells
11 to 13 in x and 3 to 5 in y (x 1.375 to 1.75 m, y 0.375 to 0.75 m). Its
bottom face is the v faces of row 3, columns 11 to 13, with the room below;
its top the v faces of row 6; its left the u faces of column 11, rows 3 to 5;
its right the u faces of column 14. ``SHELF_VENTS`` puts a vent of the other
kind on every face the box's vents leave out, each at 0.2 m/s over one cell
but the top's two:

- ``under_discharge``: bottom, x 1.375 to 1.5 (column 11), down into the room;
- ``under_exhaust``: bottom, x 1.625 to 1.75 (column 13), up into the shelf;
- ``left_discharge``: left, y 0.5 to 0.625 (row 4), toward -x;
- ``right_discharge``: right, y 0.5 to 0.625 (row 4), toward +x;
- ``top_exhaust``: top, x 1.5 to 1.75 (columns 12 and 13), down into the shelf.

The three discharges carry 300, 400 and 500 per cubic metre, so a budget can
tell them apart.

Each test file of a layer builds the configurations it needs from
``vent_raw``; the cases are data, not fixtures, so a test can change one key.
"""

import copy

from src.config import SimConfig
from validation.transport_cases import AIR, PARTICLES, SOLVER_BLOCK

WIDTH, HEIGHT, NX, NY = 2.0, 1.0, 16, 8
BOX = {"name": "box", "x_start": 0.75, "x_end": 1.25, "y_start": 0.0, "y_end": 0.5}
SUPPLY = {
    "type": "velocity_inlet",
    "location": "top",
    "x_start": 0.25,
    "x_end": 1.75,
    "velocity": 0.2,
}
RETURNS = {
    "return_left": {
        "type": "fixed_flow_outlet",
        "location": "bottom",
        "x_start": 0.0,
        "x_end": 0.5,
    },
    "return_right": {
        "type": "fixed_flow_outlet",
        "location": "bottom",
        "x_start": 1.5,
        "x_end": 2.0,
    },
}
DISCHARGE = {
    "type": "velocity_inlet",
    "location": "obstacle",
    "obstacle": "box",
    "face": "top",
    "x_start": 0.875,
    "x_end": 1.125,
    "velocity": 0.5,
}
INTAKE = {
    "type": "fixed_flow_outlet",
    "location": "obstacle",
    "obstacle": "box",
    "face": "left",
    "y_start": 0.125,
    "y_end": 0.375,
    "velocity": 0.5,
}
EXHAUST = {
    "type": "fixed_flow_outlet",
    "location": "obstacle",
    "obstacle": "box",
    "face": "right",
    "y_start": 0.125,
    "y_end": 0.375,
    "velocity": 0.4,
}
SHELF = {
    "name": "shelf",
    "x_start": 1.375,
    "x_end": 1.75,
    "y_start": 0.375,
    "y_end": 0.75,
}
_ON_SHELF = {"location": "obstacle", "obstacle": "shelf", "velocity": 0.2}
SHELF_VENTS = {
    "under_discharge": {
        **_ON_SHELF,
        "type": "velocity_inlet",
        "face": "bottom",
        "x_start": 1.375,
        "x_end": 1.5,
        "concentration": [300.0],
    },
    "under_exhaust": {
        **_ON_SHELF,
        "type": "fixed_flow_outlet",
        "face": "bottom",
        "x_start": 1.625,
        "x_end": 1.75,
    },
    "left_discharge": {
        **_ON_SHELF,
        "type": "velocity_inlet",
        "face": "left",
        "y_start": 0.5,
        "y_end": 0.625,
        "concentration": [400.0],
    },
    "right_discharge": {
        **_ON_SHELF,
        "type": "velocity_inlet",
        "face": "right",
        "y_start": 0.5,
        "y_end": 0.625,
        "concentration": [500.0],
    },
    "top_exhaust": {
        **_ON_SHELF,
        "type": "fixed_flow_outlet",
        "face": "top",
        "x_start": 1.5,
        "x_end": 1.75,
    },
}
INLET_TURBULENCE = {"turbulence_intensity": 0.05, "dissipation_length": 0.1}
TURBULENCE = {
    "model": "k_epsilon",
    "variant": "standard",
    "wall_treatment": "scalable_wall_functions",
    "cfl_number": 0.25,
    "advection_scheme": "upwind",
    "alpha_turbulence": 0.7,
    "max_iter": 500,
    "tol": 1.0e-10,
}
TRANSPORT = {
    "cfl_number": 0.1,
    "advection_scheme": "umist",
    "max_diffusion_iter": 500,
    "diffusion_tol": 1.0e-13,
}


def vent_raw(
    vents: dict[str, dict] | None = None,
    *,
    supply: bool = True,
    returns: bool = True,
    turbulence: bool = False,
    solver: dict | None = None,
    shelf: bool = False,
) -> dict:
    """The room's configuration mapping with the given vents.

    Parameters
    ----------
    vents : dict[str, dict], optional
        Segment name to segment, appended after the supply and the returns.
    supply, returns : bool
        Leave out the supply or the returns: a closed box whose only
        openings are vents.
    turbulence : bool
        Add the turbulence section, the stopping rule it needs, and the
        inlet keys on every velocity inlet.
    solver : dict, optional
        Keys that replace the solver block's.
    shelf : bool
        Add the shelf to the obstacles (its vents are the caller's to give,
        as ``SHELF_VENTS``).

    Returns
    -------
    dict
        A fresh mapping; the module's constants are not shared with it.
    """
    boundaries: dict[str, dict] = {}
    if supply:
        boundaries["hepa_supply"] = dict(SUPPLY)
    if returns:
        boundaries.update(copy.deepcopy(RETURNS))
    boundaries.update(copy.deepcopy(vents or {}))
    raw = {
        "domain": {"width": WIDTH, "height": HEIGHT, "nx": NX, "ny": NY},
        "fluid": dict(AIR),
        "particles": copy.deepcopy(PARTICLES),
        "solver": {**SOLVER_BLOCK, **(solver or {})},
        "transport": dict(TRANSPORT),
        "boundaries": boundaries,
        "obstacles": [dict(BOX)] + ([dict(SHELF)] if shelf else []),
        "sensors": [{"name": "centre", "x": 1.0, "y": 0.75}],
        "thresholds": {"5.0e-6": 100.0},
    }
    if turbulence:
        raw["turbulence"] = dict(TURBULENCE)
        raw["solver"]["stopping_rule"] = "error_estimate"
        for segment in boundaries.values():
            if segment["type"] == "velocity_inlet":
                segment.update(INLET_TURBULENCE)
    return raw


def vent_config(vents: dict[str, dict] | None = None, **options: object) -> SimConfig:
    """``vent_raw`` validated: the configuration a file of the same keys loads to."""
    return SimConfig.from_dict(vent_raw(vents, **options))  # type: ignore[arg-type]
