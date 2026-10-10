"""Append the results sections to the step 6 report from the generated tables.

Usage:
    python assemble46.py

Reads results/builder46/tables_all.md (tables46.py all) and the timing
records, and writes sections 5 to 8 of
docs/reports/ecr002_step6_product_coupled.md after its section 4, replacing
any earlier results sections. The prose is this file's; every table is the
generated one, verbatim, so a regenerated table replaces the old one by
running this script again.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "results" / "builder46"
REPORT = ROOT / "docs" / "reports" / "ecr002_step6_product_coupled.md"
RESULTS_HEADER = "## 5. Results (written after the runs)"


def tables() -> dict[str, str]:
    """The generated tables, keyed by tables46's mode name."""
    text = (OUT / "tables_all.md").read_text()
    parts = re.split(r"^### (\w+)\n", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def timing_rows() -> str:
    """The three timing probes as table rows."""
    rows = []
    for grid in ("40x15", "80x30", "200x75"):
        r = json.loads((OUT / f"standard_{grid}_time.json").read_text())
        rows.append(
            f"| {grid} | {r['seconds_per_outer']:.3f} | {r['inner_mean']:.0f} | "
            f"{r['k_sweeps_mean']:.1f}, {r['eps_sweeps_mean']:.1f} | "
            f"{r['stage_seconds']['pressure'] / r['seconds']:.2f}, "
            f"{r['stage_seconds']['turbulence'] / r['seconds']:.2f} | "
            f"{r['seconds_per_outer'] * 10000 / 60:.0f} |"
        )
    return "\n".join(rows)


def main() -> None:
    t = tables()
    body = f"""{RESULTS_HEADER}

### 5.1 Order, machine and cost

The timing probes ran first (20:58), three processes at once, 20 outer iterations each:

| Grid | Seconds per outer | CG per correction (mean) | k, eps sweeps (mean) | Share of wall: pressure, turbulence | Minutes at the 10,000 cap |
|---|---|---|---|---|---|
{timing_rows()}

The projection of the full set at the cap was under an hour of wall time in parallel, so the
set ran as planned. The seven matrix rows and the two inlet rows were launched together at
20:59:09 (nine processes, one BLAS thread each); the three located reruns and the three
transport marches followed as rows ended, so at most eleven processes ran at once on twelve
cores. The sum of the flow rows' wall times is 68 minutes; the longest single row was RNG on
200x75 to the cap, 21.5 minutes. The wall times in the tables were taken beside other runs.

Three deviations from section 2, none of which changes a measurement. (1) The scripts were
committed in three commits after the predictions, not one. (2) On 200x75 the source square
covers 30 cells of 0.04 m (0.048 m^2), not the 25 section 2.2 states: x = 2.60 is a cell face
there and y = 0.90 a cell centre, so the closed square takes five columns and six rows. On
80x30 it is the four cells stated (0.040 m^2) and on 40x15 two cells (0.080 m^2). The emission
is the same on every grid; the records carry the coverage. (3) The transport march ran as a
supplement on the converged rows (section 5.5) after the prompt's condition for measurement 4
failed, and the segment key of a floor bin was changed to "floor" alone before any
comparison was scored, since a floor piece's name carries its x interval, which the staircase
moves between grids (the `rescore` mode of `transport46.py`; the march was not repeated).

### 5.2 Measurement 1: the matrix

Every flow row, classified by section 2.1's rule. The readings columns give the five
conditions at the stop, (a) and (e) the rule's estimates over their scales, (b) and (d) in kg/s
per metre of depth against `mass_imbalance_tol`, (c) over the flux scale; "holds from" is the
first outer iteration from which each condition holds to the end. Rows ending in `_loc` are the
located reruns, which reproduce their originals' stops, counts and face hashes bit for bit.

{t["matrix"]}

The six baseline rows split by variant and grid in opposite directions. The standard variant
converges on 200x75 only; RNG converges on 40x15 and 80x30 only. No row diverged, grew or
raised a positivity error, and no pressure correction or k and eps solve reached its cap.

The bounded rows are of two kinds, and the bounded table below gives their tails (the last
2,000 outer iterations) with `bounded44.py`'s measures and, for the located reruns, the regions
of largest change. The standard rows on 40x15 and 80x30 sit in small cycles: their velocity
step is 1e-5 to 2e-4 of the velocity scale (the `velocity_step` reading against 0.45 m/s), the
per-cell imbalance is under `mass_imbalance_tol` throughout the tail (conditions (b), (c) and
(d) hold from the iterations the matrix gives), and only (a) and (e) refuse: the step neither
falls nor grows, so the fitted rate is not below one and the estimate is infinite. The cycles
are periodic in the residual (the period column), and the largest change sits in the shear
layer off the server rack's top corner above the gap (40x15), and at return 2's first cells and
the rack's top corner (80x30). The RNG row on 200x75 is a different kind: its velocity step is
about a tenth of the scale, its per-cell imbalance is above the tolerance by a factor of about
thirty, and its recurrence is weak at short lags and strongest at 75 iterations. Its largest
change sits in every tail iteration in the gap between the rack and the litho tool, beside
the litho tool's west face between 1.3 and 1.7 m up (the located row's cells), the shear
layer of the air turning down into return 2, and the largest nu_t change sits on the same
face a little lower.

{t["bounded"]}

### 5.3 Measurement 2: the inlet's turbulence

{t["inlet"]}

The three inlet settings on 80x30 with the standard variant: the strongest converges, the
baseline and the weakest reach the cap. The core's nu_t / nu follows the inlet's value to within
a few percent at the median in every row, and the 95th percentile sits two to fifteen times
above it where the shear layers reach into the core.

### 5.4 Measurement 3: what the converged room looks like

{t["converged_rows"]}

The figures named in the last column are beside this report, one per converged row: the
streamlines coloured by speed, nu_t / nu on a log scale and k. The check row at
`pressure_rtol` 1e-8 against the 1e-4 row on 200x75:

{t["rtol"]}

### 5.5 Measurement 4: where particles go

**Skipped by the prompt's condition.** Measurement 4 was to run only if both variants converge
on 80x30 and on 200x75. Neither grid has both: the standard variant did not converge on 80x30
and RNG did not converge on 200x75 (section 5.2). The two comparisons the prompt names, 80x30
against 200x75 with the standard variant and standard against RNG on 200x75, therefore have no
pair of converged records and are not scored; the comparative table below marks them skipped.

**The supplementary marches.** So that the instrument and the source's behaviour are known
before the next prompt, the march of section 2.2 was run on the three converged baseline rows
(RNG on 40x15 and 80x30, standard on 200x75). The tables below are what the method reports;
the two supplementary pairs are the ones those rows allow, a grid pair within RNG and a cross
pair between RNG on 80x30 and standard on 200x75, and neither is a pair the prompt asked for.

{t["transport"]}

{t["compare"]}

What the tables show. The source sits in return 2's capture zone: every march reaches its
steady rule within about a minute of simulated time, the removal rate equals the emission to
the rule's tolerance, and the outflow carries all but a part in ten thousand (5 micrometres) or
a part in a million (0.5 micrometres) of it. The deposition that does occur lies on the gap's
floor beside return 2 and on the two faces that bound the gap; on 40x15 the rack's east face
takes most of the 0.5 micrometre deposition, on the finer grids the litho tool's west face
and the floor piece east of the return. The four sensors read nothing: their largest value is
1e-12 per cubic metre on 80x30 and 1e-27 on 200x75 against a source-cell concentration of
1e4 to 1e5, so the sensor orders in the tables are the scheme's tails, not transport, and
their agreement or disagreement between rows means nothing. The concentration pictures
(`ecr002_step6_<run>_concentration.png`) show the plume confined to the gap below the source.

## 6. Predictions against the measurement

| Prediction | Measured |
|---|---|
| (a) The standard variant converges on all three grids | **Fails.** It converges on 200x75 (3,421) and reaches the cap on 40x15 and 80x30, bounded in small cycles (section 5.2) |
| (b) RNG converges on 40x15 and 80x30; 200x75 no prediction | **Holds** on the two grids predicted (694 and 1,407). On 200x75 it is bounded at the cap |
| (c) The core's median nu_t / nu at the baseline inlet between 10 and 100, about 16 from the inlet alone | **Holds.** The converged table gives 16 to 17 on every converged baseline row, the inlet's value; the tops' production shows in the 95th percentile, not the median |
| (d) First-node y+ on 200x75 between about 5 and 100, below 11.53 near the tops' stagnation points and in slow corners | **Holds in range, fails in place.** The median is 65 and the range 2.8 to 271, wider than predicted at both ends; the share below 11.53 is 4.5% of the nodes, all on the domain walls and the obstacle sides, none on the equipment tops (the by-kind shares in the converged table) |
| (e) `pressure_rtol` 1e-4 and 1e-8 stop at the same outer count on 200x75 | **Holds** to the iteration, 3,421, with the faces within 1.1e-11 m/s and nu_t within 1.3e-10 relative (the rtol table) |
| (f) 5 micrometres: the largest deposition on the floor nearest the source on both grids and both variants; 0.5 micrometres: the sensors keep their order between 80x30 and 200x75 | **Not scored as asked** (section 5.5). On the rows that converged the 5 micrometre hotspot is the gap's floor beside return 2 on every grid, and the 0.5 micrometre sensor order agrees between RNG 80x30 and standard 200x75, but the sensors read the scheme's tails, so the agreement carries no meaning |

Builder's predictions (section 3.2): (a) held (bounded on 200x75 was wrong in direction: it
converged there and stalled on the coarse grids); (b) failed, RNG does not follow the standard
variant grid for grid but mirrors it; (c) held, nearer 16; (d) the largest y+ is at return 1's
corner, not above 70 only but 271, and the share below the floor is 4.5%, below the fifth I
gave, and on the walls and sides rather than the tops and ceiling; (e) held to the iteration,
better than the ten I allowed; (f) the 5 micrometre hotspot held, the sensor swap was not
measurable.

## 7. What this implies

Stated as the prompt asks, without deciding.

**The coupled field converges the fine grid, with the standard variant.** Section 3.1's first
outcome (the aid before anything else) does not apply as written: on 200x75, the grid the plan
scores VAL-018 on, the standard variant converges by the rule in 3,421 outer iterations and
about ten minutes, where step 5's frozen Z2 field stayed bounded. The question the coarse grids
raise is a different one. The standard rows on 40x15 and 80x30 are converged in every reading
but the rule's two estimates: their velocity step is 1e-5 to 1e-4 of the scale, steady, and
continuity holds to the tolerance. They are small limit cycles of the kind ADR-012 C's note
found at `cfl_number` 0.5 on the Couette channel, here at 0.25, located at the rack's top
corner and return 2's first cells. Whether the rule should stop on such a row (a bound on the
step in m/s beside the estimate, as the velocity-step rule once was), whether `cfl_number`
below 0.25 or a different `alpha_turbulence` removes the cycle, or whether the cycle is left
as a coarse-grid finding since the product grid converges, is a question for Alex.

**RNG on 200x75 is not that case.** Its velocity step is a tenth of the scale and its per-cell
imbalance thirty times the tolerance, a real oscillation of the flow the ten-sweep iteration
circles. Section 3.1's first outcome applies to RNG on the product grid: the aid of step 0's
ranking (continuation in viscosity, pseudo-transient continuation), or the decision that
VAL-018 runs the standard variant and RNG stays a coarse-grid comparison until VAL-020 tells
them apart.

**The inlet's turbulence decides convergence on 80x30.** The strongest setting converges the
standard variant where the baseline and the weakest do not, and the core's eddy viscosity is
the inlet's at every setting. The supply's intensity and dissipation length are design inputs
with no measurement behind them (ADR-012 C); whether the product configuration's values should
be chosen for the room's physics or for the iteration's convergence is a question the
sensitivity pair of step 8 was meant to inform, and this measurement says the two cannot be
separated on 80x30.

**The product's question cannot be asked at this source.** The operator in the gap above
return 2 emits into the return's capture zone; everything leaves through the floor, and the
sensors read the scheme's tails. The comparative criterion of VAL-018 needs a source whose
plume reaches the sensors, or sensors placed where a plume goes: on the tops, in the gap, at
the returns. Whether the source moves (to the aisle before the door, or above an equipment
top), whether the sensors move, or whether the criterion is scored on deposition alone, is a
question for Alex before step 8. The deposition instrument itself is ready: the per-surface
rates sum to the budget's to the march's tolerance, and the 0.2 m bins compare across grids
once keyed by surface kind.

**What the check row settles.** `pressure_rtol` 1e-4 reproduces 1e-8's count to the iteration
under the coupled solve, with the faces and the eddy viscosity equal to rounding, so step 5's
finding holds with the field changing every iteration; the 1e-4 row costs 22% fewer CG
iterations per correction. ADR-013 decision 3's amendment stands on this row.

## 8. What this does not settle

- Whether the standard rows' cycles on 40x15 and 80x30 go away at a smaller `cfl_number` or
  another `alpha_turbulence`: one setting was run.
- Whether RNG's oscillation on 200x75 is the iteration's or the flow's: located at the litho
  tool's west face, not probed.
- Where particles go once the source is outside a return's capture zone: the supplementary
  marches say only that this source is inside one.
- The strain overshoot beside the walls (ADR-012 C's note) in the room: k on the equipment
  tops is not compared with anything.
- The transport march's Courant number: the configuration's 0.1 was used; the fixed point
  does not depend on it, and a larger value would only shorten the march.
"""
    text = REPORT.read_text()
    head = text.split(RESULTS_HEADER)[0].rstrip() + "\n\n"
    REPORT.write_text(head + body)
    print(f"wrote sections 5 to 8 ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
