"""Tests for the three layouts of VAL-018's comparative check (ECR-002 step 8, prompt 56).

L0 is the product room as the default file states it. L1 closes floor
return 3 and shares its flow between returns 2 and 4, the total extract
unchanged. L2 puts the HEPA supply over the left half of the ceiling only at
the same total supply flow. Each is a configuration file beside the default,
loaded the same way, and differs from it only where the layout says; the
checks here derive every expected flow from the configured ranges and the
mesh's own face widths, not from the boundary layer's resolved values, so a
layout file that quietly changed another key, or a share that did not add
up, would fail. Both exact grids (160x60 and 320x120) are checked, since the
balance holds on the discrete widths.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.boundary_registry import FIXED_FLOW_OUTLET
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import Mesh
from src.staggered import allocate_fields
from tests.test_fixed_flow_product import _segment_faces, _supply_faces

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
LAYOUTS = {
    "L0": CONFIG_DIR / "clean_room_default.yaml",
    "L1": CONFIG_DIR / "clean_room_l1_return3_closed.yaml",
    "L2": CONFIG_DIR / "clean_room_l2_left_supply.yaml",
}
EXACT_GRIDS = [pytest.param(160, 60, id="160x60"), pytest.param(320, 120, id="320x120")]
# Every stated position is a multiple of this, so both exact grids put it on
# a face (docs/reports/ecr002_step6_comparative.md, section 5.1).
POSITION_UNIT = 0.05
ROUNDING = 1e-12


def _raw(layout: str) -> dict:
    return yaml.safe_load(LAYOUTS[layout].read_text(encoding="utf-8"))


def _room(layout: str, nx: int, ny: int) -> tuple[SimConfig, Mesh, StaggeredBoundary]:
    raw = _raw(layout)
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    return config, mesh, StaggeredBoundary(mesh, config)


def _outlet_flows(
    config: SimConfig, mesh: Mesh, bc: StaggeredBoundary
) -> dict[str, float]:
    """Each fixed-flow outlet's volumetric flow: its resolved velocity times its discrete width."""
    out = {}
    for name, spec in config.boundaries.items():
        if spec.type != FIXED_FLOW_OUTLET:
            continue
        _, mask, widths = _segment_faces(config, mesh, name)
        out[name] = bc.fixed_flow_velocities()[name] * float(widths[mask].sum())
    return out


def _positions(raw: dict) -> list[tuple[str, float]]:
    """Every x and y position the configuration states, named."""
    out = []
    for name, seg in raw["boundaries"].items():
        for key in ("x_start", "x_end", "y_start", "y_end"):
            if key in seg:
                out.append((f"{name}.{key}", float(seg[key])))
    for o in raw["obstacles"]:
        for key in ("x_start", "x_end", "y_start", "y_end"):
            out.append((f"{o['name']}.{key}", float(o[key])))
    for s in raw["sensors"]:
        out.append((f"{s['name']}.x", float(s["x"])))
        out.append((f"{s['name']}.y", float(s["y"])))
    return out


@pytest.mark.unit
class TestLayoutFiles:
    """What the two layout files say, against the default."""

    def test_l1_differs_from_the_default_only_at_returns_1_and_3(self) -> None:
        """Return 3 gone, return 1 stating the default's share; nothing else moved."""
        base, l1 = _raw("L0"), _raw("L1")
        expected = yaml.safe_load(yaml.safe_dump(base))
        del expected["boundaries"]["floor_return_3"]
        expected["boundaries"]["floor_return_1"]["velocity"] = 2.7 / 2.2
        assert l1 == expected
        assert list(l1["boundaries"]) == [
            "hepa_supply",
            "door",
            "floor_return_1",
            "floor_return_2",
            "floor_return_4",
            "hood_exhaust",
        ]

    def test_l2_differs_from_the_default_only_at_the_supply(self) -> None:
        """The supply ends at 4.0 m at twice the velocity; nothing else moved."""
        base, l2 = _raw("L0"), _raw("L2")
        expected = yaml.safe_load(yaml.safe_dump(base))
        expected["boundaries"]["hepa_supply"]["x_end"] = 4.0
        expected["boundaries"]["hepa_supply"]["velocity"] = 0.9
        assert l2 == expected
        supply = l2["boundaries"]["hepa_supply"]
        base_supply = base["boundaries"]["hepa_supply"]
        assert supply["velocity"] * (
            supply["x_end"] - supply["x_start"]
        ) == pytest.approx(
            base_supply["velocity"] * (base_supply["x_end"] - base_supply["x_start"]),
            rel=ROUNDING,
        )

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_every_stated_position_is_a_multiple_of_the_exact_grids_unit(
        self, layout: str
    ) -> None:
        """So 160x60 and 320x120 put every opening, edge and sensor on a face."""
        positions = _positions(_raw(layout))
        # 38 in L0 and L2; L1 states two fewer, return 3's ends.
        assert len(positions) == (36 if layout == "L1" else 38)
        for name, value in positions:
            nearest = round(value / POSITION_UNIT) * POSITION_UNIT
            assert abs(value - nearest) < 1e-9, (layout, name, value)

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_every_layout_loads_as_the_product_configuration(self, layout: str) -> None:
        """The solver and turbulence keys of the default, on the product grid."""
        config = SimConfig(LAYOUTS[layout])
        assert (config.nx, config.ny) == (320, 120)
        assert config.turbulence is not None
        assert config.turbulence.variant == "standard"
        assert config.stopping_rule == "error_estimate"
        assert config.momentum_sweeps == 10
        assert config.max_simple_iter == 10000


@pytest.mark.integration
@pytest.mark.parametrize(("nx", "ny"), EXACT_GRIDS)
class TestLayoutFlows:
    """The flows each layout resolves on the mesh, against the default's."""

    def test_l1_keeps_return_1_and_shares_return_3_between_2_and_4(
        self, nx: int, ny: int
    ) -> None:
        """Return 1's flow is L0's; returns 2 and 4 take L0's 2, 3 and 4 at one velocity."""
        c0, m0, b0 = _room("L0", nx, ny)
        c1, m1, b1 = _room("L1", nx, ny)
        flows0, flows1 = _outlet_flows(c0, m0, b0), _outlet_flows(c1, m1, b1)
        assert "floor_return_3" not in flows1
        v0, v1 = b0.fixed_flow_velocities(), b1.fixed_flow_velocities()
        assert v1["floor_return_1"] == pytest.approx(v0["floor_return_1"], rel=ROUNDING)
        assert flows1["floor_return_1"] == pytest.approx(
            flows0["floor_return_1"], rel=ROUNDING
        )
        assert v1["floor_return_2"] == v1["floor_return_4"]
        assert v1["floor_return_2"] > v0["floor_return_2"]
        shared = flows1["floor_return_2"] + flows1["floor_return_4"]
        assert shared == pytest.approx(
            flows0["floor_return_2"]
            + flows0["floor_return_3"]
            + flows0["floor_return_4"],
            rel=ROUNDING,
        )
        assert flows1["hood_exhaust"] == flows0["hood_exhaust"]
        assert sum(flows1.values()) == pytest.approx(sum(flows0.values()), rel=ROUNDING)
        assert sum(flows1.values()) == pytest.approx(
            b1.get_total_inlet_flux(), rel=ROUNDING
        )

    def test_l2_keeps_the_total_supply_and_every_outlet_flow(
        self, nx: int, ny: int
    ) -> None:
        """Half the ceiling at twice the velocity: the same inflow, the same shares."""
        c0, m0, b0 = _room("L0", nx, ny)
        c2, m2, b2 = _room("L2", nx, ny)
        assert b2.get_total_inlet_flux() == pytest.approx(
            b0.get_total_inlet_flux(), rel=ROUNDING
        )
        mask, widths = _supply_faces(c2, m2)
        assert float(widths[mask].sum()) == pytest.approx(3.5, rel=ROUNDING)
        assert np.all(m2.xc[mask] > 0.5) and np.all(m2.xc[mask] < 4.0)
        u, v, _ = allocate_fields(m2)
        u.fill(3.0)
        v.fill(3.0)
        b2.apply_normal_velocity(u, v)
        assert np.all(v[-1, mask] == -0.9)
        assert np.all(v[-1, ~mask] == 0.0)
        v0, v2 = b0.fixed_flow_velocities(), b2.fixed_flow_velocities()
        assert list(v2) == list(v0)
        for name in v0:
            assert v2[name] == pytest.approx(v0[name], rel=ROUNDING)
        assert _outlet_flows(c2, m2, b2) == pytest.approx(
            _outlet_flows(c0, m0, b0), rel=ROUNDING
        )
