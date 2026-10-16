# ECR-002 Step 7: VAL-021 Feasibility, No Case Fits

**Date:** 2026-10-15
**Tree:** branch `feature/ecr002-val021-concentration` from main at d1866b9 (PR 82 merged). One commit,
this file. No code, no configuration, no run: item 1 of prompt 57 ends the work here.
**Outcome:** none of the three candidates fits. The recommendation is to close VAL-021 as a documented
gap (section 5). The case, the runs and the comparison (items 2 to 5) were not built.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". No threshold is set.

## 1. The question and the conditions

Is the transport model's turbulent spreading of particles right in trend against a measurement? A case
fits if all four hold (prompt 57, item 1):

- C1. The flow is ventilation-driven and isothermal or close to it (the solver has no buoyancy).
- C2. The geometry has a plane where a 2D slice is a defensible model, and the argument is stated.
- C3. The inlet, outlets, particle size and source are stated.
- C4. Concentration is measured at stated points or along stated lines in that plane, readable from the
  paper to a stated accuracy.

## 2. Prediction (a)

Prediction (a), that at least one candidate fits, is **false**. Predictions (b) and (c) are conditional
on a case and were not tested.

## 3. The candidates

Files fetched 2026-10-15 into `results/builder57/src/` (untracked). All three were obtainable.

| # | Source | Obtained from | SHA-256 | C1 | C2 | C3 | C4 |
|---|--------|---------------|---------|----|----|----|----|
| 1 | Zhang and Chen (2006), Atmos Environ 40(18):3396-3408 | engineering.purdue.edu/~yanchen/paper/2006-3.pdf (author copy, 20 pages) | f0f5656253f4e7f7... | fails | fails | holds | fails |
| 2 | Murakami, Kato, Nagano, Tanaka (1992), ASHRAE Trans 98(1):82-97 | aivc.org/sites/default/files/airbase_5635.pdf (preprint scan, 16 pages, OCR) | 9d1eb1f31f5bd279... | holds | fails | holds | fails |
| 3 | Lai and Chen (2007), Atmos Environ 41(25):5249-5256 | pmc.ncbi.nlm.nih.gov/articles/PMC7108307 (full text) | 45268fca7cc00dbc... (html) | holds | fails | holds | fails |

The prompt's DOI for candidate 3 (10.1016/j.atmosenv.2006.05.088) is the one the PMC record carries.

### 3.1 Candidate 1: Zhang and Chen (2006)

The paper has two kinds of data. Its own measurements are in a 4.91 x 2.44 x 4.31 m chamber with
underfloor supply, four heated human simulators (100 W each, 31.6 C surfaces), six ceiling lamps (64 W
each), supply 20 C against walls at 25 C (Table 2). Concentration is measured on six vertical poles
(Fig. 7), at a stated particle size (0.7 um DEHS) and source (0.3 m above the floor, S1 or S2).

- C1 fails. The flow is a displacement room carried by plumes; the paper says so ("thermal-fluid
  conditions", "vertical temperature stratification is an important characteristic"). The solver has no
  buoyancy, so the flow itself would differ, not only the particles.
- C2 fails. The chamber is three-dimensional with four manikins; no plane is a plane of symmetry or of
  two-dimensional flow.
- C4 would hold for these poles (they are points on lines), but only with C1 failing the case stays out.

Its isothermal data are Murakami's, redrawn as grey contours (its Figs. 3 and 4); the paper itself says
"these studies only reported concentration profiles in an interpolated contour form making a point to
point comparison difficult". The paper has no isothermal case of its own, so candidate 1 reduces to
candidate 2.

### 3.2 Candidate 2: Murakami et al. (1992)

Obtained, which settles the question the prompt left open about access. Isothermal (supply 1.0 m/s, all
variants), 0.31 um polystyrene (settling velocity 4.5e-6 m/s, so the particles are a passive scalar to a
first approximation, as the paper's appendix shows), source and outlets stated. C1 and C3 hold. Two
rooms carry concentration data:

- The full-scale room, 3.29 x 5.85 x 2.80 m, two ceiling supplies of 570 x 570 mm, four exhausts of
  640 x 250 mm low on the end walls, 0.64 m^3/s (Figs. 4 to 6, 8). The centre section through the
  supplies is plotted, for source heights 0.25, 0.8 and 1.6 m and source positions near the wall and
  under a jet.
- The appendix's one-fifth model, 1.05 x 0.77 x 0.49 m, one supply and one exhaust of 70 x 70 mm,
  supply velocity 2 and 0.5 m/s (Figs. 13 to 16).

**C4 fails.** Every concentration result is a shaded contour map with the same legend: five bands with
edges 0, 0.4, 0.8, 1.6 and 3.2 (dimensionless concentration, the local value over the exhaust average).
No table gives concentration at a point. Reading a point off the map gives a band, so the stated
accuracy is one band: a factor of 2 in every band above 0.4, and the lowest and highest bands are open
(0 to 0.4, above 3.2). The only point-level numbers in the paper are the correlation coefficient 0.87
between ethylene and 4.5 um particles (Fig. 17), which compares two measurements with each other, not
a measurement with a simulation. Zhang and Chen's later paper (2007, Fig. 3) captions its pole profiles
"measurement data were extracted from the original work by Murakami", but the figure plots only
simulated profiles (Eulerian and four Lagrangian sample sizes) and tabulates no measured values.
Digitising bands would give a measurement whose resolution is the whole range that prediction (b)
asks about (computed-to-measured ratios scattering "by a factor of two or more"): the data could not
resolve it.

**C2 fails.** The argument, from the paper's own geometry:

1. The supplies are square nozzles (570 x 570 mm; 70 x 70 mm in the model), the sources are points. A
   2D slice has to turn the nozzle into a plane slot and the point into a line source across the room's
   depth.
2. Keeping the flow per unit depth and the supply velocity (1 m/s) fixes the slot: 0.64 m^3/s over the
   3.29 m depth is 0.195 m^2/s, so two slots of about 0.097 m. The jet momentum then matches the room's
   (both 0.64 m^4/s^2), but the jet is a plane jet with 0.097 m width falling 2.8 m, about 29 slot
   widths, against a square jet 0.57 m wide falling about 4.9 widths, barely past its potential core.
   By the self-similar plane-jet law (centreline velocity about 2.4 U0 sqrt(b/x), an estimate, not a run)
   the 2D jet reaches the floor at about 0.45 m/s, not 1 m/s. Keeping the slot width at 0.57 m instead
   gives 0.17 m/s at equal flow. Either choice changes the flow that carries the particles, which is the
   quantity this check is meant to hold fixed.
3. A line source across 3.29 m puts the whole generation rate into one 2D plane, so the concentration
   near the source differs by an amount no 2D slice can calibrate, and the dimensionless values near the
   source are the ones the contour maps resolve best.
4. The paper itself reports a 3D disagreement: in the section view (its Fig. 3) "a distinct downward
   flow is observed in the center area in the case of the simulation", and not in the experiment, and
   the authors attribute the weaker agreement of the diffusion fields at 1.6 m source height to a supply
   velocity profile and turbulence that "cannot be precisely matched for the simulations and the
   experiments".

### 3.3 Candidate 3: Lai and Chen (2007)

The PMC full text is a model-to-model comparison: a new Eulerian drift-flux model against a modified
Lagrangian model in a 0.8 x 0.4 x 0.4 m chamber, 0.04 x 0.04 m square inlet and outlet, RNG k-epsilon,
inlet velocities 0.225 and 0.45 m/s, particles of 1 and 7 um. The text has no measured concentration
(no occurrence of "measured", "validation" or a data comparison; the single reference to experiment
points to Chen et al. (2006), Atmos Environ 40:357-367, which this work did not read). Its Table 1
compares deposition fractions between grids.

- C1 holds (isothermal, ventilation-driven).
- C2 fails for the reason in 3.2: a 3D chamber with square 40 mm ports.
- C3 holds.
- C4 fails: no measured data in this paper. The measurements the paper leans on are in Chen et al.
  (2006) and a deposition comparison would, in any case, test the near-wall deposition model rather
  than the turbulent spreading VAL-021 is about.

Chen et al. (2006) is a lead outside the three candidates the prompt named. It is a chamber of the same
class (3D, square ports), so its C2 would fail on the same argument; it has not been read.

## 4. What the check showed

- Of the three, only Murakami's isothermal rooms meet C1, and the data in them cannot meet C4 or be
  modelled in 2D (C2). The two failures are independent: with point data, the 2D argument still fails;
  with a 2D geometry, the band data would still not resolve a factor-of-2 scatter.
- Prediction (a) assumed a 2D slice of one of these 3D rooms would be defensible. The project note of
  2026-10-09 (no 2D concentration dataset exists) already warned that the check could only be a trend
  check of a slice. The slice is not defensible for these rooms (section 3.2).

## 5. What this implies (questions for Alex)

1. Close VAL-021 as a documented gap, as this report recommends, or look beyond the three named
   candidates (for example Chen et al. 2006, or a 2D-by-construction experiment such as a long
   plane-jet-driven chamber)? A search for a 2D concentration dataset was not part of item 1 and was not
   made.
2. If closed: the last unchecked link of the chain stays the turbulent spreading of particles against a
   measurement. Is it enough to record that in the limitations, or should the gap carry a stated
   uncertainty on the transport results?
3. Murakami's ethylene tracer runs (Figs. 13 and 16) are gas, not particle, data, and share the C2 and
   C4 failures above. Does the gas case change the answer to question 1? On this evidence it does not.

## 6. Record notes for the orchestrator

To apply after both branches merge. Dated 2026-10-15 (the machine's local date); the orchestrator may
re-date on applying.

**ECR-002, section 9 (acceptance criteria), item 14 (VAL-021).** Replace the "OPEN" status with:
"2026-10-15 (prompt 57): feasibility item run on the three named candidates; none fits (report
`docs/reports/ecr002_val021_feasibility.md`). Zhang and Chen (2006) own data are non-isothermal
underfloor with heated manikins; Murakami et al. (1992) is isothermal but published as five-band
contour maps (edges 0, 0.4, 0.8, 1.6, 3.2) from square-nozzle 3D rooms with point sources, which no
2D slice reproduces (the equivalent plane jet reaches the floor at about 0.45 m/s against 1 m/s);
Lai and Chen (2007) carries no measured concentration. Recommended: VAL-021 closed as a documented
gap, threshold N/A. Decision: Alex."

**ECR-002, Document History.** One row: "2026-10-15 | Prompt 57: VAL-021 feasibility, no case fits;
recommendation to close as a documented gap | Alex Moroz-Smietana" (the author line is the
orchestrator's to set).

**ADR-012, section G, the VAL-021 paragraph** (line "VAL-021, a trend-level concentration check"):
append "2026-10-15: feasibility found no fitting case among the named data; see
`docs/reports/ecr002_val021_feasibility.md`."

**PROJECT_PLAN.** In the validation table after the VAL-020 row, add VAL-021 with status "Gap: no
measurement fits a 2D slice (feasibility report, 2026-10-15); recommended closed, Alex to decide"; add
a "Step 7, VAL-021" row beside the Step 7, VAL-020 row of the ECR-002 steps table with the same status;
add a change-log row dated 2026-10-15 (prompt 57, branch feature/ecr002-val021-concentration). No
deliverable of the plan moves.

**STATUS.** In the ECR-002 paragraph after VAL-020, one sentence: "VAL-021's feasibility item found
no case that fits (Murakami's data are factor-of-two contour bands from 3D square-nozzle rooms; Zhang
and Chen's own are non-isothermal; Lai and Chen has no measured data); recommended closed as a
documented gap." Do not restate the report's table.

**SYSTEM.md.** No change: no module, interface, requirement or validation entry changes. If the
validation list names VAL-021 as planned, mark it "closed, gap" once Alex decides.
