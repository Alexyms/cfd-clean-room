# ADR-015: Alert Monitoring and Sensor Placement: Spreading Checked Both Ways, a Read-Only Monitor, Arrival Maps

## Status
Proposed. Written 2026-10-29 (prompt 70, branch `phase5/design`), before the Phase 5 build,
from the solver and the scenario machinery as ADR-010 to ADR-014 record them, the Phase 4 gate
report (`docs/reports/phase4_scenarios_report.md`, its decision of 2026-10-29 and its tables
S1, B1 to B3 and G1), ADR-014 sections C and H, SYSTEM.md's REQ-A01 to A06, and the probes
under `docs/reports/probe70/`, whose records are in `results/builder70/` (untracked, this
machine) and whose tables are `docs/reports/probe70/tables70.md`, generated from the records
by `tables70.py`, whose `--check` regenerates and compares. Three decisions were taken by Alex
on 2026-10-29 and are recorded below as given; the rest are put to Alex as a picture first,
then ranked options with their consequences, then the section carrying the detail, as
ADR-011 and ADR-014 did. The document gains a planned-against-built table at the Phase 5
gate. Bracketed numbers point at the sources at the end; "T1" to "T5" are the tables of
`tables70.md`. Nothing under `src/`, `tests/` or `configs/` changes in this pass.

## Decisions already taken (Alex, 2026-10-29), recorded as given

1. **A design pass first**, this one.
2. **The spreading condition is checked both ways:** a validation row of turbulent spreading
   against literature (a plane jet's spreading rate), and a sensitivity sweep of the scenarios
   to the two assumed inputs that set the plumes' width, the supply's turbulence intensity
   (with its dissipation length) and the turbulent Schmidt number. No detection result is
   reported before both are in. This is condition 3 of the Phase 4 gate.
3. **Placement by arrival maps:** during a scenario's march, per class and per non-SOLID cell,
   the first time the cell exceeds its threshold (the forward counterpart of ADR-014 C's local
   recovery field). Any sensor layout is then scored by lookup without rerunning, and the best
   positions are searched on top.

## Decisions for Alex
Each decision is a picture first, then the options ranked with their consequence, then the
section that carries the detail. The ranking is the builder's, from the records and the probes.

**A. The spreading validation row, VAL-024 (section A).**
*Revised 2026-10-29 (prompt 70b).* Alex's decision of 2026-10-29, on review 70 B2: VAL-024 is
the classic experiment of the breach's own mechanism, a line source in uniform decaying grid
turbulence, not a plane jet. Recorded as given; the picture and the options below are the
builder's reading of what that case tests.
*Picture.* The breach plume falls in air moving at its own speed, so nothing shears its edges;
what spreads it is the supply's own turbulence, decaying as it falls. A heated wire across a
wind tunnel downstream of a grid is that flow in a laboratory: Warhaft (1984) measured the
plume's mean temperature profile at stations downstream of a source at 52 mesh lengths in a
2.54 cm grid's turbulence at 7 m/s, intensity 2.2% at the source [11, 13]. The one case tests
two things: how the model's k and eps decay with no production (standard k-epsilon's exponent
1.087 against the measured 1.16 to 1.4 [14, 13]), and how wide its gradient-diffusion plume is
against the measured one, in the near field, where a real plume grows as sigma_v t and
gradient diffusion is known to overstate it, and in the far field, where both grow as a
diffusion. The breach plume reaches the gap sensors at half a Lagrangian time scale of the
supply's turbulence, the near field, and the floor returns at about three, so the row's
near-field stations are the ones the sensors depend on. The plane jet the first version
proposed tests shear-made turbulence instead, and review 70 B2 found its closed domain's outer
stream decelerating by a third or more across the fit range, so its records are kept as the
reason and the jet is the alternative not taken (section A's last paragraph).
*Options, ranked.*
(1) Warhaft's single line source at 52 M, as section A sets it: uniform inflow carrying the
source station's k and eps, walls 25 cm from a plume under 3 cm wide, a fixed-flow outlet, no
pressure outlet; the width read as the second moment of the profile; the reference the Anand
and Pope (1983) curve with Warhaft's parameters that Viswanathan and Pope plot [13] and the
Wilson and Sawford review says agrees with the data in the far field [15], the primary points
digitised at step 0.
(2) Stapountzis et al. (1986) [12], the companion experiment with a fuller description of the
temperature field's structure: its parameters were not readable in this pass, so it is the
cross-check once the paper is fetched.
(3) The plane jet: not taken, for the reasons above.
*Recommendation.* (1). The row is `val024_560x100` with `val024_280x50` (the probe's grid)
reported beside it; threshold proposed and OPEN: the far-field ratio at 50 and 100 M within
20% of one and the solver's decay exponent within 0.1 of the model's 1.087, the near-field
ratio reported with the cell's source size beside it. The width clause verifies no requirement
as written; REQ-T14 is proposed for it (section G). The probe's results against predictions
(c) and (d) are in the probe section.

**B. The sensitivity sweep (section B).**
*Picture.* Two inputs set how wide a plume the model makes, and both are assumptions. The
supply's turbulence intensity I and dissipation length l_e set k and eps at the inlet, hence
the eddy viscosity the air carries down from the ceiling (nu_t = C_mu sqrt(k) l_e at the
inlet, k = 1.5 (I U)^2 in the specification's convention, ADR-012 C) and so the plume's
turbulent diffusivity nu_t / Sc_t where nothing else makes turbulence, which is the zone
between the ceiling and the equipment tops where the sensors sit. The product's 5% and 0.1 m
were chosen for ECR-002's measurements, not measured in a room. HEPA-supplied unidirectional
flow is low in turbulence by design, but the sources that give a number give it below the
filter face in the room (under 2% within 0.5 m of a fan filter unit in a simulated room [7],
0% to 2% in the occupied zone and 3% to 6% near the returns in a recent CFD study validated
on velocities [6]), not at the face the solver's inlet represents; the one full-scale
measurement found reports intensity but its values were not readable [8]. The range is
therefore stated as an assumption: I from 2% to 10% and l_e from 0.05 to 0.3 m, the design
range ADR-012 D already used, whose corners put the supply's nu_t between 5e-5 and 1.5e-3
m^2/s (T2), and Sc_t from 0.2 to 1.3, VAL-018's literature range [ADR-012 source 7]. A change
of I or l_e changes the flow state (k, eps, nu_t and through them the velocity) and needs a
flow re-solve; a change of Sc_t changes the transport only and reuses the state
(`--flows-from`). The probe (T2 to T5) ran the filter breach on 160x60 at five arms: the file's
values, 10% at 0.1 m (the top of the intensity range, prediction (b)), the corner of largest
nu_t (10%, 0.3 m), the corner of smallest (2%, 0.05 m), and Sc_t 0.2 on the control's state.
*Revised 2026-10-29 (prompt 70b; review 70 B3).* What the sweep reads is geometric. A sensor
reading many orders below its threshold is the numerical tail of a sharp front, set by the
grid more than by any input: at the file's inputs the gate's 320x120 breach reads every
sensor 11 to 16 decades lower than the 160x60 control does (T3 of the gate report against
the probe's records), while the whole input range moves a sensor by 9 decades on one grid.
The observable is therefore the distance from each sensor's cell to the plume's threshold
contour, in metres and in cells, on both grids, with the contour's own grid dependence
(T6): on 160x60 the two gap sensors stand 0.28 to 0.35 m (5.7 to 7.1 cells) and 0.35 to
0.45 m (7.1 to 8.9 cells) outside the contour across the five arms, the door and hood
sensors 1.3 to 1.6 m; on 320x120 at the file's inputs 0.39 m (15.6 cells), 0.47 m (18.8
cells), 1.60 and 1.43 m. The coarse grid's contour lies about one coarse cell farther out
than the fine grid's (the width on the 2.5 m row 1.05 m against 0.98 m), so a coarse-grid
distance is a one-sided bound: a sensor outside the contour on 160x60 is farther outside it
on 320x120. A reading far below threshold is reported as "below the resolved plume", never
as a number.
*Options, ranked.*
(1) The sweep on 160x60 for every arm and every event scenario (the breach, the door, the dust
release), reading per sensor and arm the distance from the sensor's cell to the plume's
threshold contour in metres and in cells (T6's reading), per sensor row the plume's width at
the threshold level and at ten times it (T4's), and the contour's grid dependence from the
two extreme arms repeated on 320x120 for every scenario. The reason for 160x60: the question
is geometric and comparative (how far each sensor stands outside the contour, and how far the
contour moves between arms), the coarse contour bounds the fine one from outside (the
picture), the room as it is stops by the rule on 160x60 (ECR-002 step 8), every sensor and
opening sits on faces of both grids, and a run costs a fraction of the fine grid's (T2 and
the march walls, section B). Any sensor the contour reaches on 160x60 at any arm is rerun on
320x120 at that arm before a detection result is stated. The cost of the five arms on the
breach is the probe's; three scenarios at five arms is 15 runs, concurrent five at a time.
(2) The sweep on 320x120 throughout: the plume's edge is resolved as the gate's snapshots
resolve it, at about eight times the cost per run (the gate's breach run took 66 minutes on
the product grid, table M1 of the gate report).
(3) The breach alone, the scenario whose plume is nearest a sensor row: cheapest; the door and
the dust release carry their own plumes and the door's a second flow state.
*Recommendation.* (1), with the Schmidt arms reusing each arm's flow state. The report is
`docs/reports/phase5_sensitivity.md` with tables generated from the run files; no detection
result is reported before it and A are in (decision 2).

**C. The monitor (section C).**
*Picture.* The march already reads every sensor's cell every step and keeps the readings for
the run file; a clean-out's tracker already tests "below threshold" every step. The monitor
is the module that owns that reading, the thresholds and the alert, and the march calls it
where it now builds the readings list. It reads and never writes (REQ-A06): the march hands it
fields it has already made read-only. REQ-A01's "within one timestep" then holds by
construction, which leaves the question of what VAL-009 tests.
*Options, ranked.*
(1) The sensor reads the cell containing it, instantaneously, by ADR-014 C's rule (the cell
below or to the left of a sensor on a face): the one reading the clean-out's record already
uses, so Phase 4's recovery times and Phase 5's latencies are the same instrument, and T7's
time error column applies to both. An alert is raised at the first step whose reading is at or
above the class's threshold ("not below", as the recovery field counts it) and cleared at the
first step below it afterwards; the record carries both times and the peak between them.
(2) A sampled area: the mean over a box of stated size around the sensor. It smooths a cell's
noise, but the plumes' edges are a few cells wide (the gate's snapshots), so a box wider than
the edge reads low by construction, and the clean-out's instrument would differ from the
event scenarios'.
(3) A counter's sample time: the cell reading averaged over a stated `sample_time`, since a
particle counter reports a count over a sample (ISO 14644-1 asks at least a minute per sample
for classification, and monitoring counters report every few seconds to a minute [9]). It
changes every latency by about half the sample time, the same amount for every layout.
*Recommendation.* (1), with the counter's sample time stated in the latency report as an
instrument delay added beside the computed latency, not modelled in the field, until a counter
is specified (its value OPEN). VAL-009 holds by construction; what it tests (section C): that
the alert's step is the step of the first reading at or above threshold, exactly, on planted
step series (a reading equal to the threshold alerts, one below does not) and on a scenario
run whose source fills a sensor's cell, that the march calls the monitor after every step (the
run file's step series has no gap), and that the monitor leaves the fields bitwise unchanged.
REQ-A01 is reworded accordingly (G).

**D. The arrival maps and the placement search (section D).**
*Picture.* No sensor of the file's four reads any of the three event scenarios near its
threshold (the gate report's table S1). The probe's arrival maps (T5, at frame resolution from
the run files) say where a sensor would have: the breached panel's air falls onto the litho
tool's top and leaves by the gaps beside it, so the cells reached lie under the panel and in
the gaps, and the sensors at 2.5 m above the floor beside the tool stand in clean supply air
above the plume. The map is the placement instrument: once a run holds it, any layout is
scored by looking up its cells, and the best positions are a search over the map, no rerun.
*Options, ranked.*
(1) Per class and non-SOLID cell, the first step time at or above the threshold, updated every
step by a tracker beside the recovery tracker, carried in the march checkpoint's named
arrays (no format change) so a restart continues it bitwise, written to `/arrival/` at the end
of the run; a frame-resolution reader for run files that predate it (the Phase 4 and probe B
files). Latency at a cell is arrival minus the onset of the event that carries the
contamination. A layout is scored over scenarios and classes by three stated rules, worst
case, mean and coverage within a stated time, and ranked by the worst case with the mean as
the tie-break. Candidates are the non-SOLID cells within a stated standoff of a mountable
surface (walls, ceiling, obstacle tops and sides), an assumption the configuration states;
the free optimum over every non-SOLID cell is reported beside it as the bound. The search is
exact for one sensor (the argmin), exhaustive over candidate pairs for two, exact for the
worst-case rule at any count by a threshold sweep over the scenario-class pairs each cell
covers, and greedy with a one-sensor exchange for the mean rule above two; NumPy only.
(2) The map from the stored frames only, no tracker: the output cadence (0.36 s on the
product grid) sets the latency's resolution, which the output cadence should not (the
reason prompt 64 moved the recovery field into the march).
(3) The search by rerunning each candidate layout: the lookup makes it unnecessary.
*Recommendation.* (1). What the result can claim (section D): a layout's latencies are
comparative between layouts on one grid at the file's inputs, with B's spread stated beside
them; the resolution is the step, the accuracy the time error (a few percent, T7) and the
plume's position to a cell or two; the 2D slice, the steady extreme and the spreading limit
stand as ADR-014 H states them.

**E. The gate's latency report (section E).**
*Picture.* The plan asks for "all scenarios with at least two sensor configurations".
*Options, ranked.* (1) The file's four sensors and the searched four under the worst-case rule
over the three event scenarios and five classes, with the free-optimum curve for one to four
sensors as the bound, and each layout's clean-out recovery time beside its latencies (the
clearing time, ADR-014 C's companion of REQ-A04). (2) The file's four and the searched four
only. (3) Several hand-placed layouts.
*Recommendation.* (1), the tables generated from the run files by a script with a check mode.

**F. The Phase 4 conditions: order and cost (section F).**
*Picture.* Condition 3 is this design's A and B. Condition 2 is one run. Condition 1 is a
probe of its own.
*Options, ranked.* (1) Condition 3 first (steps 0 and 1 of H), condition 2 in step 1 beside
the sweep (a 160x60 clean-out, under an hour), condition 1 before Phase 6 as a prompt of its
own designed from the records, after the Phase 5 gate. (2) Condition 1 first: it bears on no
Phase 5 number (the residual is 1e-6 of the supply, the thresholds 1e-2 of the inlet value).
*Recommendation.* (1).

**G. Requirements review (section G).**
*Picture.* REQ-A01 holds by construction, A03 and A05 are not testable as written, A02, A04
and A06 are.
*Recommendation.* The rewordings of section G, three of six, put to Alex: A01 as the
monitor's contract (read every step, alert at the first step at or above threshold), A03 with
the ISO 14644-1 class formula as the testable form, A05 as scoring and searching under stated
rules in place of "optimal".

**H. Modules, build order and the coverage debt (section H).**
*Picture.* Two new modules, `monitor.py` and `placement.py`, two run-file groups, one hook in
the march; the Phase 4 coverage fall paid in the first step.
*Recommendation.* Six steps, section H: 0 the coverage debt and VAL-024; 1 the sweep and the
clean-out's third grid; 2 the monitor and VAL-009; 3 the arrival tracker, its record and
restart, REQ-A04; 4 placement, REQ-A05; 5 the reference scenarios rerun with arrival maps, the
latency report and the gate.

## Context
Phase 4 closed on 2026-10-29 as a CONDITIONAL PASS with three conditions (the gate report,
"The decision"): the budget residual's mechanism found before Phase 6's finer grids; a third
grid of the clean-out before the free-area mean's ratio or the local recovery field is quoted
as a number of the room; and a turbulent-spreading sensitivity check before Phase 5 reports
any detection result. The third is the one that decides whether Phase 5's results mean
anything. Table S1 of the gate report shows every sensor of the file's four at least 1e-19
times its threshold away from it in the three event scenarios (the breach's nearest reading
is 3.3e-24 of threshold at above_gap_1), and the snapshots show the plumes' threshold contours
a few cells from their cores. Whether that is the room's unidirectional downflow carrying the
plume past the sensors, or the model spreading plumes less than the room would, is Phase 3's
carried limit: VAL-021 closed as a documented gap because no published concentration
measurement fits a 2D slice (`docs/reports/ecr002_val021_feasibility.md`), so the transport's
turbulent spreading is unvalidated against measurement.

What Phase 5 builds on. The march (`time_integration.py`, ADR-014 A, C, D) steps every class
at one step, reads every sensor's cell every step into the run file's `/transport/sensors/`,
updates a clean-out's recovery tracker every step (the last time each cell was at or above
threshold, the per-sensor and free-area 100:1 times, the within-spec stretches), carries the
tracker's arrays in the march checkpoint and restarts bitwise. The run file holds the merged
configuration, the flow states, the frames, budgets and sensor readings, and `/recovery/`. The
thresholds are a mapping by particle size in the configuration (ISO 14644-1 Class 5 values in
the product file), the sensors a list of named positions, a sensor reading the cell containing
it (ADR-014 C's rule, `time_integration.sensor_cells`). SYSTEM.md section 4 carries a planned
`AlertMonitor` contract of three methods from the development plan, which this design
replaces.

## A. The spreading validation row, VAL-024

*Revised 2026-10-29 (prompt 70b; Alex's decision of the same date on review 70 B2).* The
first version of this section proposed a plane jet; its probes and review 70's reading of
them are kept at the end of this section as the reason the case changed.

**The mechanism the row has to test.** The breach plume falls in air moving at its own speed:
the supply is uniform across the panel and its neighbours, so there is no shear at the plume's
edges and nothing makes new turbulence there. What spreads the plume between the ceiling and
the equipment is the turbulence the HEPA supply carries down with it, decaying as it falls.
That is the classic experiment of a line source (a point in this 2D section) in decaying grid
turbulence: a heated wire across a wind tunnel downstream of a grid, the mean temperature
profile measured at stations downstream, its width against distance or flight time. A plane
jet tests shear-made turbulence instead, and in a closed domain its outer stream decays (review
70 B2), so it answers a different question.

**The sources.** Warhaft (1984, JFM 144) [11] and Stapountzis, Sawford, Hunt and Britter (1986,
JFM 165) [12] are the primary measurements; neither page was readable in this pass (Cambridge
Core gives the abstracts only, and Warhaft's abstract carries no number but a wire-spacing
range). Viswanathan and Pope (2008, Phys. Fluids 20) [13], open access, models Warhaft's single
line source and consolidates his parameters in its table I, read here from its text: grid mesh
M = 2.54 cm; mean speed U = 7 m/s (Re_M = U M / nu about 11,800 in air); the wire at
x_o / M = 20, 52 and 60 from the grid, source sizes 1.27e-4 and 2.5e-5 m; the velocity standard
deviations at one mesh length sigma_u 2.44 m/s and sigma_v = sigma_w 2.27 m/s, decaying as
(x / M)^(-m / 2) with m = 1.4 (Sawford's refit of Warhaft's transverse variance, the paper
says; the measured grid-turbulence exponents in the literature run from about 1.16 to 1.37
[14]); molecular diffusivity 2.1e-5 m^2/s. For the plume's width the paper plots (its figure 4,
source at 52 M) Anand and Pope's analytic displacement variance for the Langevin model with
C_0 = 2.1 in that decaying turbulence, its equations 11 to 15: sigma_p^2 = sigma_o^2 + 2 kappa t
+ Delta_o^2(t), Delta_o^2 = 2 sigma_v^2(t_o) t_o^2 [(1 + t / t_o)^(r - s) / (r (r - s)) +
(1 + t / t_o)^(-s) / (r s) - 1 / (s (r - s))], r = (m / 2)(3 C_0 / 2 - 1) + 1, s = (m / 2)(3 C_0 / 2
+ 1) - 1, t_o = x_o / U the flight time from the grid to the source; the review of Wilson and
Sawford (1996) [15] states (its page 196, quoted under source 15) that Anand and Pope
(1983) find this theory in good agreement with Warhaft's and Stapountzis's data in the far
field with C_0 = 2.1. The row's reference curve is therefore that theory with
Warhaft's parameters, labelled as such; the primary data points of Warhaft's figures are to be
digitised at step 0 from the JFM paper, which this machine could not fetch, and the reference
curve checked against them before the threshold is scored. At the source station the formula
gives sigma_v = 0.143 m/s and sigma_u = 0.154 m/s (an intensity of 2.2%), k_o = 0.0322 m^2/s^2,
eps_o = m k_o U / x_o = 0.239 m^2/s^3, k / eps = 0.135 s, and the Lagrangian time scale
T_L = (4 / (3 C_0)) k / eps = 0.086 s; the reference width grows from 4.7 mm (0.19 M) at
10 M downstream (t = 0.42 T_L, where sigma is still nearly sigma_v t) to 25 mm (0.98 M) at
100 M (t = 4.2 T_L, where sigma^2 grows as t^(2 - m)).

**What the model is tested on, two things in one case.** First, the decay of k and eps in a
uniform stream with no production: standard k-epsilon's solution is k = k_o (1 + t / t_v)^(-n)
with n = 1 / (C_e2 - 1) = 1.087 and t_v = k_o / ((C_e2 - 1) eps_o), against the measured grid
exponent (1.4 in the refit the reference curve uses; 1.16 to 1.37 in the grid-turbulence
literature [14]). The model's turbulence therefore decays more slowly than the tunnel's, which
is one known reason for a far-field width that reads high. Second, the plume's width against
the reference: gradient diffusion (D_t = nu_t / Sc_t, REQ-T13) grows sigma^2 as 2 D_t t from
the start, while a real plume grows as sigma_v t until t is of order T_L and only then as a
diffusion; so the model is expected wider than measured in the near field (t under T_L) and
right, or close, in the far field. Where the breach plume sits on that axis: the supply's k / eps
at the file's 5% and 0.1 m is 3.6 s, T_L about 2.3 s by the same formula, and the plume's
flight to the gap sensors' height (0.5 m at 0.45 m/s) is 1.1 s, t / T_L about 0.5, the near
field; its fall to the floor returns is about 7 s, t / T_L about 3, the transition. The row's
near-field stations (5 and 10 M, t / T_L 0.2 and 0.4) are the ones that bear on the sensors,
and the far-field ones (50 and 100 M) on the plume at the returns.

**The case** (`docs/reports/probe70/line70.py`): a domain 2.8 m long (110 M) and 0.5 m high,
uniform inflow at U over the whole left edge carrying the source station's k and eps (the
inlet's `turbulence_intensity` sqrt(k_o / 1.5) / U = 0.0209 and `dissipation_length`
k_o^(3/2) / eps_o = 0.0242 m), no-slip walls top and bottom 25 cm from the centreline, so far
from a plume under 3 cm wide that their shear layers (a few centimetres thick by the end) do
not reach it and the stream the plume sits in stays uniform, and a fixed-flow outlet over the
whole right edge stating no velocity, so the domain has no pressure outlet. The source is a
unit rate in one cell on the centreline two cells from the inlet; the plume's width is the
second moment of each column's profile about its own mean (a Gaussian's standard deviation,
the experiments' fit). The cell imposes a source size of dx / sqrt(12), 2.9 mm on the coarse
grid, which is the near field's floor on that grid (the reference's own near-field width is set
by molecular diffusion, 3.9 mm at 100 M); the cell is not the wire. The class is 0.1 um, whose
Brownian diffusivity is negligible, so the model's plume spreads by nu_t / Sc_t alone.

**The grids and the cost.** The probe's coarse grid is 280x50 (1 cm cells, 14,000 cells); the
row's is 560x100 (5 mm, 56,000 cells, about five cells across sigma at 10 M and ten at 100 M).
The cost is estimated from the coarse probe's measured outer count and per-iteration time
scaled by the cell count (the probe section), not from the plane jet's walls, which were taken
during a slowdown (S6).

**The harness rows.** `val024_280x50` and `val024_560x100`, one name per grid; a case file
`configs/validation_line_source.yaml`; `validation/line_source.py` with the reference curve,
`plume_widths` and `decay_exponent`, pinned by `tests/test_line_source.py` on a planted Gaussian
field and on the analytic decay; `validation.cases.load_line_source`. The rows report n, sigma
at the five stations against the reference and the ratio, scored by the threshold Alex sets.

**Threshold, OPEN.** Proposed: the far-field ratio sigma / sigma_ref at 50 and 100 M within
20% of one, and the decay exponent n within 0.1 of the model's own 1.087 (a check of the
solver, not of the model, since the model's exponent is known to sit below the measured
range); the near-field ratio at 5 and 10 M reported, not scored, with the cell's source size
beside it. An alternative: score against the digitised data directly once they are in.

**Trace.** The decay clause verifies REQ-S14 (the k-epsilon model as ADR-012 A names it; the
row joins VAL-015's decaying-turbulence test under its Verified By). The width clause verifies
no requirement as written: REQ-T13 states the diffusivity's formula, which the solver meets by
construction, and nothing requires the formula to spread a scalar as a measurement does.
Proposed, for Alex (section G gains the row): REQ-T14, "With the turbulence model on, the
transport solver's spreading of a passive scalar released from a line source in decaying grid
turbulence shall match the measured plume width in the far field within a stated allowance,
the near-field width reported beside it", verified by VAL-024.

**The alternative not taken: the plane jet.** The first version of this row was a plane jet in
a weak coflow, probed on 2026-10-29 (T1; `jet70.py`; `jet70.png`). Its measured spreading rate
0.100 to 0.110 for five still-air studies and 0.118 for a sixth [2, table 2], and standard
k-epsilon's 0.108 from the steady similarity solution [1] (0.103 unsteady; 0.110 and 0.117 with
the boundary-layer and the full production term [3]), are for a jet in still air or a constant
outer stream. In this solver the jet needs a closed domain with a coflow to feed the
entrainment (a pressure outlet draws air in, #61), and review 70 B2 read the records: on the
revised domains the outer stream beside the jet falls from 85% to 57% of its inlet value across
15 to 45 slot widths at 0.1 U_j on 120x100 and from 65% to 28% at 0.05 U_j on 120x200, with an
adverse pressure rise of about 0.26 Pa over the fit range, so the "weak coflow" was a
decelerating confined stream and the half-width's background was a choice (the inlet coflow as
background moves K1 from 0.0932 to 0.0798, 0.0862 to 0.0763 and 0.1057 to 0.0961, 9% to 14%).
The measured shifts, each on the arm where it was measured and not summed: 0.0932 on 120x100
and 0.0862 on 240x200 at 0.1 U_j (14% and 20% under 0.108); 0.1057 at 0.05 U_j on 120x200;
0.0973 with k and eps by UMIST at the iteration cap on 120x100. The jet's scalar spread 1.21 to
1.29 times faster than its velocity on every converged arm, against the measured 1.15 to 1.52
[2, table 7] and the constant-eddy-viscosity ratio 1.224 at Sc_t 0.7. The entrainment the
first, 30 slot-width domain could not supply: the jet's excess flux is about 0.49 U_j b
sqrt(x / b) from [2]'s constants (the centreline excess decaying as 2.28 U_j sqrt(b / x), the
half-width growing at 0.1 x, the profile integrating to 2.16 half-widths), 3.3 slot flows at
45 slot widths (2.3 entrained) and 3.8 at the outlet at 60 (2.8 entrained), against 1.45
supplied. The jet is a shear-flow test, and the line source is the breach's mechanism; the jet
stays in the records and could return as a second row if a free-shear check is wanted later,
with the outer stream's deceleration sized and the background defined as the local outer
velocity.

## B. The sensitivity sweep

**The inputs and their ranges.**

| Input | Where it enters | Range | Source or assumption | Needs a flow re-solve |
|---|---|---|---|---|
| Supply turbulence intensity I | `boundaries.hepa_supply.turbulence_intensity` (and the breach's three segments, which carry the supply's keys); k = 1.5 (I U)^2 | 2% to 10% | Assumption: ADR-012 D's design range; the sources give 0% to 2% in the occupied zone and 3% to 6% near returns of an FFU room [6], under 2% within 0.5 m of a unit in a simulated room [7], none at the filter face itself; 10% is the vents file's fan-jet value | yes |
| Supply dissipation length l_e | `dissipation_length`; eps = k^(3/2) / l_e | 0.05 to 0.3 m | Assumption: ADR-012 D's design range (a tenth of a filter unit's short side to half of it) | yes |
| Turbulent Schmidt number Sc_t | `transport.turbulent_schmidt`; D_t = nu_t / Sc_t | 0.2 to 1.3 | VAL-018's literature range (Tominaga and Stathopoulos 2007, ADR-012 source 7) | no |

The supply's inlet nu_t at the corners (T2): 5e-5 m^2/s (2%, 0.05 m) to 1.5e-3 m^2/s (10%,
0.3 m), a factor of thirty, nu_t / nu about 3 to 100. Between the ceiling and the equipment
tops nothing produces turbulence, so the plume's diffusivity in the sensors' zone is the
supply's nu_t decaying, over Sc_t.

**What is read** (revised 2026-10-29, prompt 70b, review 70 B3). Per arm, per scenario, per
class: the distance from each sensor's cell to the nearest non-SOLID cell ever at or above
the threshold, centre to centre, in metres and in cells (T6's reading), on both grids for
the extreme arms; on each sensor's row, the plume's width at the threshold level and at ten
times it, at the end of the event and the largest over the frames, and the extent of the
threshold cells (T4's); the arrival map's fraction of cells reached (T5's). A sensor's peak
reading is stated as a number only when it is within a hundredth of its threshold (the
resolved range of the front on either grid); below that it is "below the resolved plume",
because the tail's value is the grid's (11 to 16 decades between 160x60 and 320x120 at the
same inputs, T3 of the gate report against T3 here) and not the inputs'. The plume's width
and the sensors' distances are the quantities the inputs move; the map says whether any
position would have read it. The arrival times read from stored frames are bounded by the
frame spacing (T5's caption) and are not latencies; section D's in-march maps remove that
bound.

**The grid.** 160x60, with the reason given under decision B, and the two extreme arms of the
breach repeated on 320x120. The width's resolution on 160x60 is a 5 cm cell, so a width that
changes by under a cell between arms is reported as unchanged to the cell.

**The cost.** From T2 and the probe's march walls (the probe section): a 160x60 breach arm is
its flow solve plus its 120 s march. The three event scenarios at five arms are 15 runs,
concurrent five, and the two fine-grid arms are two runs at the gate's cost (66 minutes each,
concurrent).

**The report.** `docs/reports/phase5_sensitivity.md`, tables generated from the run files by
a script under `docs/reports/probe72/` with a check mode, the arms' configurations written by
a generator as `configs70.py` writes the probe's. It states, per scenario, whether any arm
brings any sensor within a stated factor of its threshold, and the plume width's range over
the arms. Decision 2's clause: no detection result before it and A are in.

## C. The monitor

**Contract.** `src/monitor.py`, replacing SYSTEM.md section 4's planned block:

```
monitor.py --> time_integration, placement; scripts/run_scenario.py
SENSOR_FACE_RTOL                                   # moved here from time_integration (re-exported there)
sensor_cells(config, mesh) -> tuple[(j, i), ...]   # moved here: ADR-014 C's rule, ValueError on SOLID
Alert: sensor (name), size_class, step_on, t_on, reading_on, step_off, t_off (None while open),
       peak, peak_t                                # frozen
AlertMonitor(config, mesh)                         # the sensors' cells, the names, the class thresholds;
                                                   # ValueError with no sensor or no threshold
    update(step, t, fields) -> tuple[Alert, ...]   # the readings of this step and the alerts it opens
                                                   # or closes; fields read, never written (REQ-A06);
                                                   # step above the last, t above the last
    readings -> np.ndarray                         # [classes, sensors] of the last update
    alerts -> tuple[Alert, ...]                    # every alert so far, open ones last
    latencies(onsets) -> np.ndarray                # [events, classes, sensors]: t_on of the first alert
                                                   # at or after each onset minus the onset, inf if none
                                                   # before the next onset or the end
    state() / restore(state)                       # the open alerts and the peaks, for the checkpoint
```

**Where it sits.** In `_Run.march`, after the step's fields are made read-only and before the
recovery tracker's update: `alerts = self.monitor.update(step, t, fields)`; the readings it
returns replace `self.readings(fields)` in the pending list, and the alerts go to the run file
at each flush as a table `/transport/alerts/` (step_on, t_on, step_off, t_off, size_class,
sensor, reading_on, peak, peak_t), replaced whole at each write as `/recovery/` is. The
monitor's `state()` joins the checkpoint's named arrays, so a restart continues an open alert.
The on_step callback and the runner's summary print the alerts opened in the run.

**The sensor model.** Decision C's option (1): the containing cell, instantaneous, ADR-014 C's
rule, the reading the clean-out's record already uses. "At or above" opens an alert and
"below" closes it, matching the recovery field's "not below". A counter's sample time is
reported as an instrument delay beside the latency (section E), OPEN.

**Thresholds per class (REQ-A03).** The `thresholds` mapping by size stays the contract. An
optional second form, `thresholds: {iso_class: N}`, computes the mapping at load by the
standard's formula C_n = 10^N (0.1 um / D)^2.08 rounded to three significant figures, which
gives the file's five Class 5 values (100,000; 10,200; 3,520; 832; 29.3 per cubic metre) and
is the testable sense of "following ISO 14644-1" [9]. The standard's 2015 table omits some
size-class pairs (no 5 um entry at Class 5, none at 0.1 and 0.3 um at Class 8); the formula
is used for them as the file does, and the loader says so in its docstring.

**VAL-009.** As written ("detection delay 0 or 1 timestep") it holds by construction when the
monitor reads every step: the alert is the step's reading. What it tests, as
`tests/test_alert_latency.py`:
- on planted step series handed to `update`, the alert's step is the step of the first
  reading at or above threshold, exactly; a reading equal to the threshold opens the alert
  and one below it does not; the alert closes at the first step below; a second crossing
  opens a second alert;
- on a scenario run (80x30, a source in a sensor's cell), the run file's alert table and
  sensor series agree: the alert's step is the first step whose stored reading is at or above
  threshold, and the step series has no gap (the march called the monitor every step);
- the fields are bitwise unchanged across `update` (the bytes compared) and their writeable
  flag is False (REQ-A06);
- a restart across an open alert reproduces the uninterrupted run's alert table bitwise.
Its criterion row: the alert step equals the crossing step (exact), the latency's resolution
is the step. REQ-A01 reworded as section G proposes.

## D. The arrival maps and the placement search

**The map.** Per class k and non-SOLID cell, `first_above[k, j, i]`, the first step time at
which C_k is at or above threshold_k; NaN until then (inf in the record for a cell never
reached, NaN in SOLID cells, as `local_recovery` is written). Updated every step by
`monitor.ArrivalTracker(thresholds, live)` beside the recovery tracker (the same `above` mask,
the recovery tracker keeps the last time, this one the first), carried in the march
checkpoint's `recovery` dict of named arrays under `first_above` (no format change:
`MarchCheckpoint.recovery` is a dict of float64 arrays by name), restored by shape as the
recovery tracker's are, so a restart continues it bitwise. Today the march passes
`recovery=None` when there is no clean-out tracker (`_Run.march`); with the arrival tracker
every tier-1 run carries the dict, the clean-out's with both trackers' arrays in it. Written at the end of the run and at
each checkpoint flush as `/arrival/` (`ArrivalRecord`: thresholds, onsets with their event
names, `first_above` [classes, ny, nx], t_final; `write_arrival` and `read_arrival` on
`RunFile`). The clean-out gets the map too; its arrival is t = 0 everywhere above threshold,
which is what the clearing-time companion needs to say nothing new.

**Run files without it.** `placement.arrival_from_frames(run_file)` reads the stored frames
and returns the map at the output cadence's resolution, flagged as such. The Phase 4 run files
and probe B's are read this way (T5), and section F's sweep can be read before step 3 is
built.

**Latency** (revised 2026-10-29, prompt 70b, review 70 S7). The onset of an event that
carries contamination is its `start` (a `replace` with a concentration key, a `source`, or
the initial condition at t = 0). The tracker keeps one map per such event: at each onset
the map for that event starts, and records per class and cell the first step time at or
after the onset at which the cell is at or above threshold, so a cell already at or above
it at the onset reads the onset itself. The latency for event e at a cell is that time minus
`start_e`; infinite if the cell is not reached before t_end, and the report says "never"
with the run's end beside it. The monitor's `latencies(onsets)` is defined the same way on
the sensors' cells: for each onset, the time of the first reading at or above threshold at
or after it, an alert open at the onset counting as the onset, so the two instruments
agree at a sensor's cell by construction and a test pins that they do. Each reference
scenario has one such event, so its latency is arrival minus 10 s (the door, the breach,
the dust) and zero for the clean-out; a scenario with several is reported per event.

**Scoring a layout.** A layout is a set S of cells. Per scenario s and class k,
L(s, k, S) = min over c in S of latency(s, k, c). Over the scenario-class pairs P (three
scenarios, five classes, fifteen pairs for the reference set):
- worst case: max over P of L; infinite when a pair is never detected;
- mean: the mean over P of min(L, T_cap(s)), T_cap the run's remaining time after the onset,
  with the count of undetected pairs beside it;
- coverage: the fraction of P with L at or below a stated `within` time.
Layouts are ranked by the worst case, ties by the mean; the report shows all three. The rules
are `placement.score(maps, onsets, cells, rule)` on the maps in memory, so scoring every
candidate single cell is one vectorised minimum.

**Candidates.** An assumption the configuration states, `placement.candidates`: `mount` from
{walls, ceiling, obstacle_tops, obstacle_sides, free} and `standoff` in metres; a candidate is
a non-SOLID cell whose centre is within the standoff of a named surface, excluding cells
adjacent to an inlet or outlet face (a sensor in the supply's or a return's face reads the
boundary, not the room). The free optimum over every non-SOLID cell is always reported beside
the mountable one, as the bound.

**The search.** `placement.search(maps, onsets, candidates, count, rule)`:
- count 1: the argmin over candidates (exact);
- count 2: exhaustive over candidate pairs (on 320x120 with a two-cell standoff about 1,800
  mountable cells, 1.6 million pairs, fifteen pairs each: seconds in NumPy);
- the worst-case rule at any count: exact by a threshold sweep over a set cover (revised
  2026-10-29, review 70 S8). Pairs reached by no candidate are removed first and reported
  as uncovered, so the worst case is over the coverable pairs. For each distinct latency T
  in increasing order, the pairs a cell covers within T form a mask of at most fifteen bits;
  the question whether `count` masks OR to all ones is settled by dynamic programming over
  the 2^15 subsets (the fewest masks covering each subset, built mask by mask over the
  distinct masks, whose number is measured and reported, not assumed), and the smallest T
  with a cover of the count is the optimum. The covers attaining it are enumerated by a
  depth-first search over the distinct masks bounded by the count, which is small, and the
  tie among them is broken by the mean. When no pair is coverable the ranking falls to the
  mean rule, and the report says so;
- the mean rule above two: greedy (add the cell that lowers the score most) followed by
  one-sensor exchanges until none improves, checked against the exhaustive answer at count 2.
NumPy only (ADR-013 decision 2: no new dependency).

**What the result can claim.** The latencies are comparative between layouts on one grid at
the file's inputs; B's spread is stated beside them. A latency's resolution is the step
(0.0036 s on the product grid at Courant 0.4), its accuracy the time error of a few percent
(T7, measured in one direction on 160x60) and the plume edge's position to a cell or two,
which at 0.45 m/s is a few hundredths of a second but at a slow lingering edge can be seconds.
The best position is the best under the model on this grid; the claim Phase 5 can make is the
one VAL-018 makes for deposition, that the ranking of layouts is stable under the grid pair
and across B's arms, stated per rule with ties, not an absolute latency. ADR-014 H's limits
stand: the 2D slice (a line source, exchanges per metre of depth), the steady extreme, the
turbulence over the tops, the spreading limit now bounded by A and B rather than open.

## E. The gate's latency report

`docs/reports/phase5_alert_report.md` with tables generated from the run files by
`docs/reports/probe76/tables76.py` (check mode):
- per scenario, class and sensor, for each configuration: latency (or never with the run's
  end), peak over threshold, time above, T7's time error column, B's width spread column;
- the configurations: the file's four sensors; the searched four under the worst-case rule
  over the three event scenarios and five classes on the mountable candidates; the free
  optimum for one to four sensors as a curve (the bound); for each layout, the clean-out's
  room recovery time and per-sensor recovery times (ADR-014 C) as the clearing companion;
- the counter's sample time stated as an additive instrument delay, OPEN;
- the maps as figures: per scenario the 0.1 um and 5 um arrival maps with the sensors drawn.

## F. The Phase 4 conditions: order and cost

| Condition | When | What it costs | What it could change |
|---|---|---|---|
| 3: the spreading check (A and B) | Steps 0 and 1 of H, before any detection result | A: T1's two solves and marches on a workstation core each (the probe section gives the walls); B: 15 coarse runs concurrent five and two fine runs (section B) | Whether the plumes' widths and the sensors' distances to the contour move with the assumed inputs: if the contour reaches a sensor's cell at an arm on 160x60, that arm is rerun on 320x120 and the report states the detection result as a range over the arms on the fine grid; if none does on either grid, the "never" is geometric and the maps carry the placement question |
| 2: a third clean-out grid | Step 1, beside the sweep (a run, no code) | One 160x60 clean-out, 240 s at Courant 0.4: about an eighth of the product grid's 2.6 hours (table M1), under an hour with its flow solve | Whether the mean's 100:1 ratio and the local recovery field can be quoted as numbers of the room: a 160x60 value between 80x30's 3.0 and 320x120's 4.8 shows a series not yet converged and the numbers stay comparative; the room recovery time (25.6 and 27.9 s) is unaffected either way |
| 1: the budget residual's mechanism | Before Phase 6's finer grids, after the Phase 5 gate, a prompt of its own | One session: a sealed-box march (VAL-014's) with a source and an interval switch at a known time on 40x15, the budget read every step, the residual's accrual against the content and its sign at the switches (the records put the sign changes near 10.5 and 25 s in the dust release and 80.4 s in the breach, and B3 rules the flow's continuity out by two orders) | The budget instrument (REQ-T05's verification) if the residual is bookkeeping between the content and the running sums or the source's integral at a switch; the scheme's inlet booking if the limited face value at an inlet face differs from the inflow the budget books. Neither moves a Phase 5 latency: the residual is 1e-6 of the supply against thresholds at 1e-2 of the inlet value |

## G. Requirements review, REQ-A01 to A06

| Requirement | What it asks | Testable as written | Proposed rewording (where needed) |
|---|---|---|---|
| REQ-A01 "detect threshold exceedance within one timestep of occurrence" | A latency bound on the instrument | No: it holds by construction when the monitor reads every step, so no test can fail it; "occurrence" is the step's reading | "The alert monitor shall read every sensor after every transport step and raise an alert at the first step whose reading is at or above the class's threshold, so that it adds no delay of its own to the detection latency, whose resolution is the step." Verified by VAL-009 as section C defines it |
| REQ-A02 sensor locations configurable in YAML | The sensors section | Yes (unit: a sensor added or moved in the mapping changes the cell read; a SOLID cell refused; the face rule) | Add the reading rule: "a sensor reads the cell containing it, the one below or to the left of a position on a face (ADR-014 C)" |
| REQ-A03 thresholds per class following ISO 14644-1 | The thresholds mapping | Partly: configurable per class is testable; "following ISO 14644-1" is not without the formula | "Contamination thresholds shall be configurable per particle size class, either as a mapping by size or as an ISO 14644-1 class number from which the loader computes the limits by the standard's formula rounded to three significant figures; the product file's Class 5 values are the test" |
| REQ-A04 latency per sensor per scenario from event onset | The report's core number | Yes once onset is defined (integration test on a coarse run) | Append: "the onset is the start of the event that carries the contamination (a replacement with a concentration, a source, or the initial condition at t = 0); a sensor never at or above threshold within the run reports no latency and the run's end" |
| REQ-A05 multiple configurations across scenarios to identify optimal placement | The placement question | No: "optimal" names no rule | "The monitor's placement module shall score any set of sensor layouts over a set of scenario run files by stated rules (worst case, mean, coverage within a stated time), rank them, and search the stated candidate positions for the layout of a stated count that is best under a rule, exact for one and two sensors and for the worst-case rule at any count over the pairs some candidate reaches, reproducible from the run files alone" |
| REQ-A06 reads but never modifies | Separation of concerns | Yes, and by test as well as review: the fields' bytes before and after `update`, the writeable flag False | Append: "verified by test: the fields are bitwise unchanged across every monitor call" |
| REQ-T14, proposed (prompt 70b, review 70 S9) | VAL-024's width clause verifies no requirement as written: REQ-T13 states the diffusivity's formula, met by construction | A new requirement, for Alex | "With the turbulence model on, the transport solver's spreading of a passive scalar released from a line source in decaying grid turbulence shall match the measured plume width in the far field within a stated allowance, the near-field width reported beside it." Verified by VAL-024; the decay clause of VAL-024 joins REQ-S14's Verified By |

## H. Modules, contracts, the build order and the coverage debt

**The module cut.**

```
monitor.py --> time_integration, placement; scripts/run_scenario.py; tests
    (section C's block)
    ArrivalTracker(thresholds, live)
        update(t, fields) -> None               # first_above where NaN and at or above
        first_above -> np.ndarray               # [classes, ny, nx], NaN until reached
        state() / restore(state)
        record(onsets, t_final) -> ArrivalRecord

placement.py --> scripts/latency_report.py (step 5); tests
    arrival_from_run(run_file) -> (maps, onsets, resolution)    # /arrival/, else from the frames, flagged
    arrival_from_frames(frames, thresholds, live) -> np.ndarray
    latency(maps, onsets) -> np.ndarray         # [events, classes, ny, nx], inf never
    score(latencies, cells, rule, *, within=None) -> Score     # worst, mean, undetected, coverage
    candidates(mesh, config, mount, standoff) -> np.ndarray     # [n, 2] cells
    search(latencies, candidates, count, rule) -> Layout        # cells, score, method, exact (bool)
    rank(layouts) -> list[Layout]

io_manager.py
    RunFile.write_alerts(rows) / read_alerts()  # /transport/alerts/, replaced whole
    RunFile.write_arrival(record) / read_arrival()   # /arrival/
    ArrivalRecord: thresholds, event_names, onsets, first_above, t_final

time_integration.py
    _Run.monitor: AlertMonitor; _Run.arrival: ArrivalTracker (every tier-1 run)
    march: update both after each step; alerts flushed with the readings; both states in the
    checkpoint's named arrays; /arrival/ written at the end
    RunRecord gains alerts (tuple[Alert, ...]) and arrival (ArrivalRecord)
    sensor_cells, SENSOR_FACE_RTOL re-exported from monitor

config.py
    thresholds: {iso_class: N} as a second form (section C)
    placement: PlacementSpec | None: mount (tuple of names), standoff (m), count, rule, within
```

Dependencies: `monitor -> config, mesh`; `placement -> config, mesh, io_manager, monitor`;
`time_integration -> monitor` (new edge); `io_manager` gains two groups and no import. The
system map (`scripts/gen_system_map.py`) regenerates section 3 at each step; SYSTEM.md
section 4's planned monitor block is replaced by section C's at acceptance and marked built
at step 2.

**The build order.** ECR-style steps, one pull request each, review and test before each:

| Step | Prompt | Builds | Tests | Validation row |
|---|---|---|---|---|
| 0 | 71 | The coverage debt: tests for the 66 uncovered statements of table Cv2 of the gate report (47 `raise` lines, 19 fallbacks and early returns), each exercised by its own test or recorded as unreachable with the reason; VAL-024's case file, loader, readers and harness rows, and Warhaft's plume widths digitised from the JFM paper | `tests/test_line_source.py`; the new rejection-path tests across `test_config.py`, `test_io_manager.py`, `test_solver_transport.py`, `test_turbulence.py`, `test_mesh.py`, `test_time_integration.py`, `test_forces.py`, `test_particles.py`, `test_staggered.py`, `test_solver_staggered.py` | VAL-024, scored by Alex's threshold |
| 1 | 72 | The sensitivity sweep (section B) as a report with its generator and tables; the clean-out's third grid (condition 2) as a run and a row in the same report | none in `src/`; the report's check mode | the sweep's tables; no row |
| 2 | 73 | `monitor.py` (AlertMonitor, Alert, sensor_cells moved), the march hook, `/transport/alerts/`, the runner's summary | `tests/test_monitor.py` (unit: thresholds by size and by class, the cell rule, open and close, the iso_class form, REQ-A06's bytes), `tests/test_alert_latency.py` | VAL-009 |
| 3 | 74 | `ArrivalTracker`, `/arrival/`, the checkpoint arrays and restart, `latencies(onsets)` and REQ-A04's per-sensor latency in `RunRecord` and the summary; `placement.arrival_from_frames` | `tests/test_monitor.py` (the tracker on planted series, restart bitwise), `tests/test_time_integration.py` (the map against the frames' map on a coarse run) | none; REQ-A04's integration test |
| 4 | 75 | `placement.py`: latency, score, candidates, search, rank; the `placement` section | `tests/test_placement.py` (exhaustive equals the sweep and the greedy on small maps, the rules on planted maps, the candidate rule), `tests/test_sensor_evaluation.py` (REQ-A05: the three event scenarios on 80x30, two layouts scored and ranked, the search's layout at least as good as the file's) | none; REQ-A05's integration test |
| 5 | 76 | The four reference scenarios rerun on 320x120 with the maps (about three hours wall, concurrent), the latency report (section E), the Phase 5 gate report with the planned-against-built table | none new | the gate's criteria |

**The coverage debt.** Phase 4's statements fell from 99.25% to 98.86% and branches from
97.13% to 96.21% (Cv1). Step 0 pays it before any Phase 5 module lands: its pull request's
coverage at or above Phase 3's two figures, measured by the same command (`pytest tests/
--cov=src --cov-branch`), so that the Phase 5 gate's "does not decrease" is measured from
Phase 4's figures with the debt already cleared. A line found unreachable is listed in the
pull request with the reason, not excluded from measurement.

## Probes (the predictions fixed in the prompt before the runs; results after)

*On the order of commits (review 70 S1).* The orchestrator's predictions (a) and (b) were
fixed in the prompt's text before any run. The ADR's copy of them, with the builder's,
was first committed in e77fd65 at 14:12 on 2026-10-29, after the first probe records
started at 10:41, so the builder's "before the runs" has no committed record that shows
it. Predictions (c) and (d) of prompt 70b were committed in 7109c8b before their probe
ran. Future design passes commit the probe section before running.

**Predictions (orchestrator, 2026-10-29).**
- (a) The code's standard k-epsilon plane-jet spreading rate falls within 15% of the
  literature's value for standard k-epsilon.
- (b) On 160x60, the filter breach at the top of the supply-intensity range still leaves every
  sensor below its threshold, with the plume's width at sensor height grown by under a factor
  of two.

**Predictions (orchestrator, 2026-10-29, prompt 70b; on the line-source row of the revised
section A, committed before its probe ran).**
- (c) The model's k decays with an exponent between 1.0 and 1.3.
- (d) The model's plume is wider than measured in the near field and within 20% of the
  measured width in the far field.

**Builder (2026-10-29, prompt 70b, before the line-source probe, scored apart).**
- (c) Holds: the model's own solution has n = 1 / (C_e2 - 1) = 1.087, and in a uniform stream
  with no production the solver has only the source terms to integrate, so the read exponent
  is expected within 0.02 of it; the walls' shear layers stay far from the centreline.
- (d) Holds in the near field (the cell's source size of 2.9 mm on the coarse grid alone makes
  the plume wider than the reference's 4.7 mm at 10 M by more than the model's gradient
  diffusion does); fails narrowly in the far field: with nu_t nearly constant along the stream
  (k^2 / eps decays as t^(1 - n)), the model's sigma at 100 M is about sqrt(2 D_t t) with D_t
  = 0.129 k^2 / eps / 0.7 at the source, about 20 mm against the reference's 25 mm, a ratio
  near 0.8, at the band's edge; the direction (narrower) is the sign of the model's smaller
  far-field diffusivity against the Langevin reference's (4 / (3 C_0)) k / eps time scale.

**Builder (2026-10-29, before the runs, scored apart).**
- (a) Holds, read against 0.108: the model with the full production term is tabulated at 0.117
  and the band reaches 0.124; the coarse slot (four cells) smears the initial shear layer,
  which moves the virtual origin and not the slope, so 240x120 is expected between 0.10 and
  0.12 and 120x60 lower by the coflow's and the grid's share. The scalar's ratio is expected
  near 1.2, inside the measured 1.15 to 1.52.
- (b) Holds, and the "never" is geometric: the supply's nu_t at 5% is about 2.4e-4 m^2/s and
  at 10% twice that, so a plume falling 0.5 m at 0.45 m/s spreads by sqrt(2 D t) of about 3 cm
  at 5% and 4 cm at 10%, under a cell of 160x60; the width at the threshold level on the
  sensors' row grows by well under two. The sensors at 2.5 m beside the litho tool stand in
  the clean supply upstream of the panel's plume, which lands on the tool's top and leaves by
  the gaps, so no arm brings them within reach; the arrival maps show the reached cells under
  the panel and in the gaps.

**Method.** `jet70.py` (T1; its docstring fixes the readings), `configs70.py` and `run70.sh`
(T2 to T5; the arms, the readings in `tables70.py`'s docstring). Wall times were taken with
six processes sharing the machine (two jet solves and four breach arms), one BLAS thread each;
the revised jet solves ran three at a time with the first fine-grid solve and the Schmidt arm.
*The slowdown (review 70 S6).* Between about 11:20 and 14:10 on 2026-10-29 the machine ran
the solver processes at a small fraction of a core (the 240x120 scalar march took 0.24 s per
step against 240x200's 0.025 s on more cells; the UMIST arm 0.67 s per outer iteration
against upwind's 0.064 s on the same grid). T1's last two flow walls (9,684 and 10,119 s)
and the 240x120 march's 9,789 s fell in that window and are not costs of the cases; the
outer counts are. The line-source probe ran after it.

*Note (2026-10-29, after the first jet version ran, before the second).* The first version's
domain (30 slot widths high, a 0.05 U_j coflow) confined the jet (section A: the coflow
supplied 1.45 slot flows against 2.8 entrained by the outlet), so the domain was revised to 50 slot widths
and 0.1 U_j and rerun on 120x100 and 240x200. The confined records stay in T1 as labelled
rows. The predictions above were written before any run and stand unchanged against the
revised runs; the builder's (a) was written for the confined grids and is scored on the
revised ones. *Second note, after the revised pair ran.* The row grid read 20% under 0.108,
so two diagnostic arms were added on the coarse resolution to apportion the gap, each
changing one input of the 120x100 arm: the coflow (0.05 U_j in a 100 slot-width domain,
120x200) and the k and eps advection scheme (UMIST in place of upwind). Their rows are T1's
last two.

**Results.** `docs/reports/probe70/tables70.md` (T1 to T5), copied at the end of this section;
the figures `docs/reports/probe70/jet70.png` (the half-widths and fits) and `arrival70.png`
(the 0.1 um arrival maps of the control and largest-viscosity arms with the sensors drawn).

*Prediction (a): FAILS on the row grid, holds on the coarse.* On the 50 slot-width domain at
0.1 U_j the velocity's K1 is 0.0932 on 120x100 (13.7% under 0.108, inside the band) and 0.0862
on 240x200 (20.2% under, outside it), both solves stopped by the rule (2,496 and 6,000 outer
iterations), both fits linear to a residual of 0.01 and 0.001 slot widths over 15 to 45. The
coflow carries most of the shortfall on the coarse resolution: 0.05 U_j in the 100 slot-width
domain reads 0.1057 (2.1% under 0.108, inside the measured 0.100 to 0.110), 4,662 outer
iterations by the rule. k and eps advected by UMIST in place of upwind on the 0.1 U_j arm
reads 0.0973, 4% above the upwind arm's 0.0932, at the iteration cap rather than by the rule
(the limited scheme on k and eps cycles, as ADR-012 C found on the room). The three shifts
are each measured on one arm against the 120x100 upwind arm and are not a decomposition of
the row grid's gap (review 70 S2): the coflow's 0.0125, the scheme's 0.0041 and the grid's
0.0070 (two to four cells across the slot) sum to more than the gap of 0.0218, and review 70
B2 found the outer stream decelerating by a third or more across the fit range on every
revised domain, so the arms are confined jets at two confinements, not a weak coflow at two
ratios (section A, the alternative not taken). The confined first version
read 0.115 (two cells across the slot) and 0.125 (four) at the iteration cap, both with the
return flow, and its scalar never settled; those rows are evidence for the domain, not
measurements of the model.

*The scalar beside the velocity.* 1.224, 1.243 and 1.288 on the three converged arms (the
scalar's K1 0.114, 0.107 and 0.136), inside the measured 1.15 to 1.52 and within 5% of the
constant-eddy-viscosity ratio 1.224 at Sc_t 0.7 (arcsech(0.5^(1 / 1.4)) over
arcsech(0.5^(1 / 2)), 1.079 / 0.881); the scalar marches reached a steady field in 6 to 15
simulated seconds. The scalar's background read near zero in both domains, so B2's
background sensitivity does not move the scalar's K1.

*Prediction (b): HOLDS, rescored on the plume's edge (prompt 70b, review 70 B3).* On 160x60
the breach at 10% intensity (0.1 m) leaves every sensor below its threshold in every class,
every reading below the resolved plume (T3), and the plume's width on the sensors' row at
2.5 m is 1.10 m against 1.05 m at the file's 5% (T4), a factor 1.05. The edge reading (T6):
at the file's values above_gap_1 stands 0.354 m (7.1 cells) outside the threshold contour
and above_gap_2 0.447 m (8.9 cells); at 10% and 0.1 m the same 0.354 m and 0.430 m (8.6
cells); at the widest arm of the range, 10% at 0.3 m (supply nu_t 99 nu, six times the
file's), 0.283 m (5.7 cells) and 0.354 m (7.1 cells), the width on the row 1.20 m (a factor
1.14); at the narrowest, 2% at 0.05 m, 0.354 and 0.447 m, the width 0.95 m; at Sc_t 0.2 on
the file's flow 0.320 and 0.391 m. The door and hood sensors stand 1.3 to 1.6 m outside at
every arm. The width at ten times the threshold moves by one cell or none. The contour's
grid dependence, from the gate's 320x120 breach at the file's inputs read the same way:
above_gap_1 0.389 m (15.6 cells), above_gap_2 0.470 m (18.8 cells), the width on the 2.5 m
row 0.98 m against 1.05 m, so the coarse contour lies about one coarse cell farther out and
the coarse distances bound the fine ones from below. Every flow state stopped by the rule
in 1,953 to 3,067 outer iterations (T2); a march of about 15,000 steps took 610 to 700 s
wall with five processes sharing the machine, before the slowdown named under Method.

*The builder's (a) and (b).* (a) was wrong on the row grid in the direction it did not expect
(the slope under the band, not the virtual origin moved); its ratio expectation of about 1.2
held. (b) held in every part, including the geometry: the arrival maps (T5, `arrival70.png`)
put the earliest reached cell under the panel's edge at 2.8 m height, by the first stored
frame after the onset (0.24 to 0.47 s after it, each the thirteenth frame of its arm, so an
upper bound by up to one frame spacing of 0.79 to 0.81 s and not a measured latency; review
70 S5), the reached cells (17% to 20% of the room's free cells) under the panel, on the
litho tool's top and in both gaps to the floor returns, and the file's four sensors outside
the reached set at every arm.

*What the results change in the design.* Decision A's row moves to the 0.05 U_j domain
(240x400, step 0), the threshold stays proposed and OPEN with the refinement trend named;
decision B's reading stands, with the width's change over the whole range (0.95 to 1.20 m
on the sensors' row) as the number the sweep will state beside every latency; decision D's
maps are the instrument the sweep already needed.

**Tables** (copied from `docs/reports/probe70/tables70.md` at the end of the pass; the generated file is the record and `tables70.py --check` compares it with the records).

#### T1. The plane jet's spreading rate (probe A, jet70.py), standard k-epsilon on the product's model and solver keys

| Case | Cells across the slot | Outer iterations | Stop | Flow wall (s) | K1 velocity | Virtual origin (slot widths) | Fit RMS residual (slot widths) | K1 scalar (Sc_t 0.7) | Scalar / velocity | Scalar march: steps, stop, wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 b high, coflow 0.05 U_j (confined), 240x120 | 4 | 15000 | max_simple_iter | 2316 | 0.1251 | 2.30 | 0.036 | 0.2315 | 1.851 | 40304, cap, 9789 |
| 30 b high, coflow 0.05 U_j (confined), 120x60 | 2 | 15000 | max_simple_iter | 768 | 0.1146 | -5.43 | 0.123 | 0.8622 | 7.525 | 20095, cap, 121 |
| 50 b high, coflow 0.1 U_j, 240x200 | 4 | 6000 | error_estimate_and_continuity | 1141 | 0.0862 | -0.75 | 0.001 | 0.1071 | 1.243 | 5817, steady, 143 |
| 50 b high, coflow 0.1 U_j, 120x100 | 2 | 2496 | error_estimate_and_continuity | 159 | 0.0932 | -4.06 | 0.011 | 0.1141 | 1.224 | 3234, steady, 31 |
| 100 b high, coflow 0.05 U_j, 120x200 (the coflow's share) | 2 | 4662 | error_estimate_and_continuity | 9684 | 0.1057 | -3.79 | 0.013 | 0.1361 | 1.288 | 7494, steady, 106 |
| 50 b high, coflow 0.1 U_j, 120x100, k and eps by UMIST (the scheme's share) | 2 | 15000 | max_simple_iter | 10119 | 0.0973 | 3.70 | 0.021 | 0.1179 | 1.212 | 2755, steady, 26 |

| Source | K1 velocity | Scalar / velocity |
|---|---|---|
| Measured, plane jets into still air (Gutmark and Wygnanski 1976; Browne et al. 1983; Thomas and Chu 1989; Thomas and Prakash 1991; Ramaprian and Chandrasekhara 1985) | 0.100 to 0.110 | - |
| Standard k-epsilon, steady RANS similarity solution (Wilcox, as tabulated by Premnath and Abraham 2005) | 0.108 | - |
| Standard k-epsilon, unsteady RANS (Magi et al., same table) | 0.103 | - |
| Standard k-epsilon, parabolic march, simplified / full production term (Chen and Kim 1987) | 0.110 / 0.117 | - |
| Measured scalar (temperature) to velocity, K1_scalar / K1_u (Stanley and Sarkar 1999, table 7: Ramaprian and Chandrasekhara, Browne et al., Jenkins and Goldschmidt, Davies et al.) | - | 1.15 to 1.52 |
| DNS, Re_h 3,000, scalar to velocity (Stanley and Sarkar 1999) | 0.093 (velocity) | 1.51 |

#### T2. The breach arms on 160x60 (probe B, run70.sh): inputs and the flow solve

| Arm | Intensity I | Dissipation length l_e (m) | Sc_t | Supply nu_t = C_mu sqrt(k) l_e (m^2/s) | nu_t / nu | Outer iterations | Stop | Flow wall (s) |
|---|---|---|---|---|---|---|---|---|
| i05 | 0.05 | 0.10 | 0.7 | 2.48e-04 | 16 | 1953 | error_estimate_and_continuity | 109 |
| i10 | 0.10 | 0.10 | 0.7 | 4.96e-04 | 33 | 2024 | error_estimate_and_continuity | 114 |
| i10_l30 | 0.10 | 0.30 | 0.7 | 1.49e-03 | 99 | 3067 | error_estimate_and_continuity | 156 |
| i02_l05 | 0.02 | 0.05 | 0.7 | 4.96e-05 | 3 | 2282 | error_estimate_and_continuity | 122 |
| sc02 | 0.05 | 0.10 | 0.2 | 2.48e-04 | 16 | 1953 | error_estimate_and_continuity | 109 |

#### T3. The sensors: peak reading over threshold and first time at or above, per arm and class

| Arm | Class (um) | Sensor | Peak over threshold | First time at or above (s) |
|---|---|---|---|---|
| i05 | 0.1 | near_door | 5.042e-21 | never |
| i05 | 0.1 | above_gap_1 | 2.107e-13 | never |
| i05 | 0.1 | above_gap_2 | 7.167e-19 | never |
| i05 | 0.1 | hood_entry | 5.549e-24 | never |
| i05 | 0.3 | near_door | 5.039e-21 | never |
| i05 | 0.3 | above_gap_1 | 2.107e-13 | never |
| i05 | 0.3 | above_gap_2 | 7.166e-19 | never |
| i05 | 0.3 | hood_entry | 5.546e-24 | never |
| i05 | 0.5 | near_door | 5.036e-21 | never |
| i05 | 0.5 | above_gap_1 | 2.105e-13 | never |
| i05 | 0.5 | above_gap_2 | 7.165e-19 | never |
| i05 | 0.5 | hood_entry | 5.543e-24 | never |
| i05 | 1 | near_door | 5.019e-21 | never |
| i05 | 1 | above_gap_1 | 2.100e-13 | never |
| i05 | 1 | above_gap_2 | 7.158e-19 | never |
| i05 | 1 | hood_entry | 5.527e-24 | never |
| i05 | 5 | near_door | 4.545e-21 | never |
| i05 | 5 | above_gap_1 | 1.960e-13 | never |
| i05 | 5 | above_gap_2 | 6.964e-19 | never |
| i05 | 5 | hood_entry | 5.072e-24 | never |
| i10 | 0.1 | near_door | 5.989e-18 | never |
| i10 | 0.1 | above_gap_1 | 1.192e-11 | never |
| i10 | 0.1 | above_gap_2 | 6.927e-15 | never |
| i10 | 0.1 | hood_entry | 1.490e-20 | never |
| i10 | 0.3 | near_door | 5.987e-18 | never |
| i10 | 0.3 | above_gap_1 | 1.191e-11 | never |
| i10 | 0.3 | above_gap_2 | 6.926e-15 | never |
| i10 | 0.3 | hood_entry | 1.489e-20 | never |
| i10 | 0.5 | near_door | 5.982e-18 | never |
| i10 | 0.5 | above_gap_1 | 1.191e-11 | never |
| i10 | 0.5 | above_gap_2 | 6.924e-15 | never |
| i10 | 0.5 | hood_entry | 1.488e-20 | never |
| i10 | 1 | near_door | 5.963e-18 | never |
| i10 | 1 | above_gap_1 | 1.188e-11 | never |
| i10 | 1 | above_gap_2 | 6.918e-15 | never |
| i10 | 1 | hood_entry | 1.483e-20 | never |
| i10 | 5 | near_door | 5.430e-18 | never |
| i10 | 5 | above_gap_1 | 1.117e-11 | never |
| i10 | 5 | above_gap_2 | 6.731e-15 | never |
| i10 | 5 | hood_entry | 1.351e-20 | never |
| i10_l30 | 0.1 | near_door | 1.144e-09 | never |
| i10_l30 | 0.1 | above_gap_1 | 7.146e-07 | never |
| i10_l30 | 0.1 | above_gap_2 | 2.285e-08 | never |
| i10_l30 | 0.1 | hood_entry | 9.204e-12 | never |
| i10_l30 | 0.3 | near_door | 1.144e-09 | never |
| i10_l30 | 0.3 | above_gap_1 | 7.145e-07 | never |
| i10_l30 | 0.3 | above_gap_2 | 2.285e-08 | never |
| i10_l30 | 0.3 | hood_entry | 9.201e-12 | never |
| i10_l30 | 0.5 | near_door | 1.143e-09 | never |
| i10_l30 | 0.5 | above_gap_1 | 7.143e-07 | never |
| i10_l30 | 0.5 | above_gap_2 | 2.285e-08 | never |
| i10_l30 | 0.5 | hood_entry | 9.196e-12 | never |
| i10_l30 | 1 | near_door | 1.140e-09 | never |
| i10_l30 | 1 | above_gap_1 | 7.132e-07 | never |
| i10_l30 | 1 | above_gap_2 | 2.282e-08 | never |
| i10_l30 | 1 | hood_entry | 9.175e-12 | never |
| i10_l30 | 5 | near_door | 1.057e-09 | never |
| i10_l30 | 5 | above_gap_1 | 6.836e-07 | never |
| i10_l30 | 5 | above_gap_2 | 2.215e-08 | never |
| i10_l30 | 5 | hood_entry | 8.557e-12 | never |
| i02_l05 | 0.1 | near_door | 2.309e-22 | never |
| i02_l05 | 0.1 | above_gap_1 | 5.174e-16 | never |
| i02_l05 | 0.1 | above_gap_2 | 6.007e-27 | never |
| i02_l05 | 0.1 | hood_entry | 2.976e-31 | never |
| i02_l05 | 0.3 | near_door | 2.289e-22 | never |
| i02_l05 | 0.3 | above_gap_1 | 5.172e-16 | never |
| i02_l05 | 0.3 | above_gap_2 | 6.005e-27 | never |
| i02_l05 | 0.3 | hood_entry | 2.974e-31 | never |
| i02_l05 | 0.5 | near_door | 2.253e-22 | never |
| i02_l05 | 0.5 | above_gap_1 | 5.169e-16 | never |
| i02_l05 | 0.5 | above_gap_2 | 6.002e-27 | never |
| i02_l05 | 0.5 | hood_entry | 2.972e-31 | never |
| i02_l05 | 1 | near_door | 2.125e-22 | never |
| i02_l05 | 1 | above_gap_1 | 5.155e-16 | never |
| i02_l05 | 1 | above_gap_2 | 5.989e-27 | never |
| i02_l05 | 1 | hood_entry | 2.961e-31 | never |
| i02_l05 | 5 | near_door | 3.443e-23 | never |
| i02_l05 | 5 | above_gap_1 | 4.748e-16 | never |
| i02_l05 | 5 | above_gap_2 | 5.614e-27 | never |
| i02_l05 | 5 | hood_entry | 2.673e-31 | never |
| sc02 | 0.1 | near_door | 1.327e-13 | never |
| sc02 | 0.1 | above_gap_1 | 2.469e-09 | never |
| sc02 | 0.1 | above_gap_2 | 1.759e-11 | never |
| sc02 | 0.1 | hood_entry | 1.421e-15 | never |
| sc02 | 0.3 | near_door | 1.326e-13 | never |
| sc02 | 0.3 | above_gap_1 | 2.468e-09 | never |
| sc02 | 0.3 | above_gap_2 | 1.759e-11 | never |
| sc02 | 0.3 | hood_entry | 1.421e-15 | never |
| sc02 | 0.5 | near_door | 1.326e-13 | never |
| sc02 | 0.5 | above_gap_1 | 2.467e-09 | never |
| sc02 | 0.5 | above_gap_2 | 1.759e-11 | never |
| sc02 | 0.5 | hood_entry | 1.420e-15 | never |
| sc02 | 1 | near_door | 1.322e-13 | never |
| sc02 | 1 | above_gap_1 | 2.463e-09 | never |
| sc02 | 1 | above_gap_2 | 1.757e-11 | never |
| sc02 | 1 | hood_entry | 1.416e-15 | never |
| sc02 | 5 | near_door | 1.219e-13 | never |
| sc02 | 5 | above_gap_1 | 2.337e-09 | never |
| sc02 | 5 | above_gap_2 | 1.710e-11 | never |
| sc02 | 5 | hood_entry | 1.316e-15 | never |

#### T4. The plume's width on the sensor rows: cells at or above the threshold (1% of the panel's air) and ten times it, times dx; at the frame nearest 70 s and the largest over the frames

| Arm | Class (um) | Row y (m) | Width at 1% at 70 s (m) | Largest width at 1% (m) | Width at 10% at 70 s (m) | Largest width at 10% (m) | Extent of the 1% cells at 70 s (m) |
|---|---|---|---|---|---|---|---|
| i05 | 0.1 | 2.50 | 1.05 | 1.05 | 0.90 | 0.90 | 3.15 to 4.20 |
| i05 | 0.1 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i05 | 0.1 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i05 | 0.1 | 1.20 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i05 | 0.3 | 2.50 | 1.05 | 1.05 | 0.90 | 0.90 | 3.15 to 4.20 |
| i05 | 0.3 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i05 | 0.3 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i05 | 0.3 | 1.20 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i05 | 0.5 | 2.50 | 1.05 | 1.05 | 0.90 | 0.90 | 3.15 to 4.20 |
| i05 | 0.5 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i05 | 0.5 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i05 | 0.5 | 1.20 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i05 | 1 | 2.50 | 1.05 | 1.05 | 0.90 | 0.90 | 3.15 to 4.20 |
| i05 | 1 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i05 | 1 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i05 | 1 | 1.20 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i05 | 5 | 2.50 | 1.00 | 1.05 | 0.90 | 0.90 | 3.20 to 4.20 |
| i05 | 5 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i05 | 5 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i05 | 5 | 1.20 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i10 | 0.1 | 2.50 | 1.10 | 1.10 | 0.90 | 0.90 | 3.15 to 4.25 |
| i10 | 0.1 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i10 | 0.1 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i10 | 0.1 | 1.20 | 1.00 | 1.00 | 0.85 | 0.85 | 2.50 to 4.80 |
| i10 | 0.3 | 2.50 | 1.10 | 1.10 | 0.90 | 0.90 | 3.15 to 4.25 |
| i10 | 0.3 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i10 | 0.3 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i10 | 0.3 | 1.20 | 1.00 | 1.00 | 0.85 | 0.85 | 2.50 to 4.80 |
| i10 | 0.5 | 2.50 | 1.10 | 1.10 | 0.90 | 0.90 | 3.15 to 4.25 |
| i10 | 0.5 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i10 | 0.5 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i10 | 0.5 | 1.20 | 1.00 | 1.00 | 0.85 | 0.85 | 2.50 to 4.80 |
| i10 | 1 | 2.50 | 1.10 | 1.10 | 0.90 | 0.90 | 3.15 to 4.25 |
| i10 | 1 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i10 | 1 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i10 | 1 | 1.20 | 1.00 | 1.00 | 0.85 | 0.85 | 2.50 to 4.80 |
| i10 | 5 | 2.50 | 1.10 | 1.10 | 0.90 | 0.90 | 3.15 to 4.25 |
| i10 | 5 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i10 | 5 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i10 | 5 | 1.20 | 1.00 | 1.00 | 0.85 | 0.85 | 2.50 to 4.80 |
| i10_l30 | 0.1 | 2.50 | 1.20 | 1.20 | 1.00 | 1.00 | 3.10 to 4.30 |
| i10_l30 | 0.1 | 2.05 | 2.10 | 2.10 | 1.90 | 1.90 | 2.65 to 4.75 |
| i10_l30 | 0.1 | 1.50 | 1.05 | 1.05 | 0.85 | 0.85 | 2.50 to 4.85 |
| i10_l30 | 0.1 | 1.20 | 1.10 | 1.10 | 0.90 | 0.90 | 2.45 to 4.85 |
| i10_l30 | 0.3 | 2.50 | 1.20 | 1.20 | 1.00 | 1.00 | 3.10 to 4.30 |
| i10_l30 | 0.3 | 2.05 | 2.10 | 2.10 | 1.90 | 1.90 | 2.65 to 4.75 |
| i10_l30 | 0.3 | 1.50 | 1.05 | 1.05 | 0.85 | 0.85 | 2.50 to 4.85 |
| i10_l30 | 0.3 | 1.20 | 1.10 | 1.10 | 0.90 | 0.90 | 2.45 to 4.85 |
| i10_l30 | 0.5 | 2.50 | 1.20 | 1.20 | 1.00 | 1.00 | 3.10 to 4.30 |
| i10_l30 | 0.5 | 2.05 | 2.10 | 2.10 | 1.90 | 1.90 | 2.65 to 4.75 |
| i10_l30 | 0.5 | 1.50 | 1.05 | 1.05 | 0.85 | 0.85 | 2.50 to 4.85 |
| i10_l30 | 0.5 | 1.20 | 1.10 | 1.10 | 0.90 | 0.90 | 2.45 to 4.85 |
| i10_l30 | 1 | 2.50 | 1.20 | 1.20 | 1.00 | 1.00 | 3.10 to 4.30 |
| i10_l30 | 1 | 2.05 | 2.10 | 2.10 | 1.90 | 1.90 | 2.65 to 4.75 |
| i10_l30 | 1 | 1.50 | 1.05 | 1.05 | 0.85 | 0.85 | 2.50 to 4.85 |
| i10_l30 | 1 | 1.20 | 1.10 | 1.10 | 0.90 | 0.90 | 2.45 to 4.85 |
| i10_l30 | 5 | 2.50 | 1.20 | 1.20 | 1.00 | 1.00 | 3.10 to 4.30 |
| i10_l30 | 5 | 2.05 | 2.10 | 2.10 | 1.90 | 1.90 | 2.65 to 4.75 |
| i10_l30 | 5 | 1.50 | 1.05 | 1.05 | 0.85 | 0.85 | 2.50 to 4.85 |
| i10_l30 | 5 | 1.20 | 1.10 | 1.10 | 0.90 | 0.90 | 2.45 to 4.85 |
| i02_l05 | 0.1 | 2.50 | 0.95 | 0.95 | 0.90 | 0.90 | 3.20 to 4.15 |
| i02_l05 | 0.1 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i02_l05 | 0.1 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i02_l05 | 0.1 | 1.20 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i02_l05 | 0.3 | 2.50 | 0.95 | 0.95 | 0.90 | 0.90 | 3.20 to 4.15 |
| i02_l05 | 0.3 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i02_l05 | 0.3 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i02_l05 | 0.3 | 1.20 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i02_l05 | 0.5 | 2.50 | 0.95 | 0.95 | 0.90 | 0.90 | 3.20 to 4.15 |
| i02_l05 | 0.5 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i02_l05 | 0.5 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i02_l05 | 0.5 | 1.20 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i02_l05 | 1 | 2.50 | 0.95 | 0.95 | 0.90 | 0.90 | 3.20 to 4.15 |
| i02_l05 | 1 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i02_l05 | 1 | 1.50 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i02_l05 | 1 | 1.20 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| i02_l05 | 5 | 2.50 | 0.95 | 0.95 | 0.90 | 0.90 | 3.20 to 4.15 |
| i02_l05 | 5 | 2.05 | 1.90 | 1.90 | 1.80 | 1.80 | 2.75 to 4.65 |
| i02_l05 | 5 | 1.50 | 0.90 | 0.95 | 0.85 | 0.85 | 2.60 to 4.80 |
| i02_l05 | 5 | 1.20 | 0.95 | 0.95 | 0.85 | 0.85 | 2.55 to 4.80 |
| sc02 | 0.1 | 2.50 | 1.10 | 1.10 | 1.00 | 1.00 | 3.15 to 4.25 |
| sc02 | 0.1 | 2.05 | 2.00 | 2.00 | 1.80 | 1.80 | 2.70 to 4.70 |
| sc02 | 0.1 | 1.50 | 1.00 | 1.05 | 0.85 | 0.85 | 2.55 to 4.85 |
| sc02 | 0.1 | 1.20 | 1.05 | 1.05 | 0.90 | 0.90 | 2.50 to 4.85 |
| sc02 | 0.3 | 2.50 | 1.10 | 1.10 | 1.00 | 1.00 | 3.15 to 4.25 |
| sc02 | 0.3 | 2.05 | 2.00 | 2.00 | 1.80 | 1.80 | 2.70 to 4.70 |
| sc02 | 0.3 | 1.50 | 1.00 | 1.05 | 0.85 | 0.85 | 2.55 to 4.85 |
| sc02 | 0.3 | 1.20 | 1.05 | 1.05 | 0.90 | 0.90 | 2.50 to 4.85 |
| sc02 | 0.5 | 2.50 | 1.10 | 1.10 | 1.00 | 1.00 | 3.15 to 4.25 |
| sc02 | 0.5 | 2.05 | 2.00 | 2.00 | 1.80 | 1.80 | 2.70 to 4.70 |
| sc02 | 0.5 | 1.50 | 1.00 | 1.05 | 0.85 | 0.85 | 2.55 to 4.85 |
| sc02 | 0.5 | 1.20 | 1.05 | 1.05 | 0.90 | 0.90 | 2.50 to 4.85 |
| sc02 | 1 | 2.50 | 1.10 | 1.10 | 1.00 | 1.00 | 3.15 to 4.25 |
| sc02 | 1 | 2.05 | 2.00 | 2.00 | 1.80 | 1.80 | 2.70 to 4.70 |
| sc02 | 1 | 1.50 | 1.00 | 1.05 | 0.85 | 0.85 | 2.55 to 4.85 |
| sc02 | 1 | 1.20 | 1.05 | 1.05 | 0.90 | 0.90 | 2.50 to 4.85 |
| sc02 | 5 | 2.50 | 1.10 | 1.10 | 1.00 | 1.00 | 3.15 to 4.25 |
| sc02 | 5 | 2.05 | 2.00 | 2.00 | 1.80 | 1.80 | 2.70 to 4.70 |
| sc02 | 5 | 1.50 | 1.00 | 1.00 | 0.85 | 0.85 | 2.55 to 4.85 |
| sc02 | 5 | 1.20 | 1.05 | 1.05 | 0.90 | 0.90 | 2.50 to 4.85 |

#### T5. The arrival map at frame resolution (every 100 steps): cells reached, the earliest arrival, and the map at the file's sensors

| Arm | Class (um) | Non-SOLID cells reached by the end | Earliest arrival: cell centre (x, y) m, time (s), latency from 10 s | Map at near_door, above_gap_1, above_gap_2, hood_entry (s) |
|---|---|---|---|---|
| i05 | 0.1 | 17.17% | (3.475, 2.825), 10.28, 0.28 | never, never, never, never |
| i05 | 0.3 | 17.15% | (3.475, 2.825), 10.28, 0.28 | never, never, never, never |
| i05 | 0.5 | 17.15% | (3.475, 2.825), 10.28, 0.28 | never, never, never, never |
| i05 | 1 | 17.15% | (3.475, 2.825), 10.28, 0.28 | never, never, never, never |
| i05 | 5 | 17.15% | (3.475, 2.825), 10.28, 0.28 | never, never, never, never |
| i10 | 0.1 | 17.49% | (3.525, 2.775), 10.31, 0.31 | never, never, never, never |
| i10 | 0.3 | 17.49% | (3.525, 2.775), 10.31, 0.31 | never, never, never, never |
| i10 | 0.5 | 17.49% | (3.525, 2.775), 10.31, 0.31 | never, never, never, never |
| i10 | 1 | 17.49% | (3.525, 2.775), 10.31, 0.31 | never, never, never, never |
| i10 | 5 | 17.44% | (3.475, 2.775), 10.31, 0.31 | never, never, never, never |
| i10_l30 | 0.1 | 20.09% | (3.475, 2.675), 10.47, 0.47 | never, never, never, never |
| i10_l30 | 0.3 | 20.09% | (3.475, 2.675), 10.47, 0.47 | never, never, never, never |
| i10_l30 | 0.5 | 20.09% | (3.475, 2.675), 10.47, 0.47 | never, never, never, never |
| i10_l30 | 1 | 20.09% | (3.475, 2.675), 10.47, 0.47 | never, never, never, never |
| i10_l30 | 5 | 20.03% | (3.475, 2.675), 10.47, 0.47 | never, never, never, never |
| i02_l05 | 0.1 | 16.58% | (3.525, 2.825), 10.24, 0.24 | never, never, never, never |
| i02_l05 | 0.3 | 16.58% | (3.525, 2.825), 10.24, 0.24 | never, never, never, never |
| i02_l05 | 0.5 | 16.58% | (3.525, 2.825), 10.24, 0.24 | never, never, never, never |
| i02_l05 | 1 | 16.58% | (3.525, 2.825), 10.24, 0.24 | never, never, never, never |
| i02_l05 | 5 | 16.58% | (3.525, 2.825), 10.24, 0.24 | never, never, never, never |
| sc02 | 0.1 | 18.69% | (3.525, 2.775), 10.28, 0.28 | never, never, never, never |
| sc02 | 0.3 | 18.69% | (3.525, 2.775), 10.28, 0.28 | never, never, never, never |
| sc02 | 0.5 | 18.69% | (3.525, 2.775), 10.28, 0.28 | never, never, never, never |
| sc02 | 1 | 18.69% | (3.525, 2.775), 10.28, 0.28 | never, never, never, never |
| sc02 | 5 | 18.69% | (3.525, 2.775), 10.28, 0.28 | never, never, never, never |


## Consequences

- Phase 5 reports no detection result before VAL-024 and the sweep are in (decision 2), so
  the first two steps carry no monitor code; the monitor lands at step 2.
- The monitor is the march's reading instrument, so Phase 4's recovery times and Phase 5's
  latencies are one instrument with one time error column.
- The arrival map joins the recovery field as the second per-cell output of a march, carried
  in the checkpoint and the run file; a placement answer is a lookup over run files and never
  a rerun.
- REQ-A01, A03 and A05 are reworded to testable forms if Alex accepts G; VAL-009's criterion
  becomes the step identity.
- The line source becomes the solver's first scalar-spreading validation row, the breach's
  own mechanism, and the spreading limit is bounded rather than open; VAL-021 stays closed.

## Alternatives considered

- A monitor that samples a volume or integrates over a counter's sample time (C's options 2
  and 3): deferred, the sample time to be stated beside the latency.
- The map from the frames only (D's option 2) and the search by rerunning (D's option 3):
  rejected for the output cadence's resolution and the cost.
- The plane jet (A's option 3, the first version's row): not taken, a shear-flow test whose
  closed domain confines the jet (review 70 B2); its records stay in section A.
- Condition 1 first (F's option 2): rejected, it bears on no Phase 5 number.

## Sources

1. Premnath, K. N. and Abraham, J. (2005), "Discrete lattice BGK Boltzmann equation
   computations of transient incompressible turbulent jets", arXiv:physics/0503208 (Int. J.
   Mod. Phys. C); table 1 collects the plane jet's spreading rates: measured 0.100 (Gutmark
   and Wygnanski 1976), 0.104 (Browne et al. 1983), 0.110 (Thomas and Chu 1989), the range
   0.100 to 0.110; standard k-epsilon 0.108 (steady RANS, Wilcox) and 0.103 (unsteady RANS,
   Magi et al.); k-omega 0.101; DNS 0.092; LES 0.094 and 0.106. Fetched 2026-10-29,
   https://arxiv.org/pdf/physics/0503208, SHA-256 63d9d028...7595e6, 26 pages.
2. Stanley, S. A. and Sarkar, S. (1999), "A study of the flowfield evolution and mixing in a
   planar turbulent jet using direct numerical simulation", LBNL-43005 preprint (J. Fluid Mech.
   450, 2002, with Mellado); tables 2 and 7 (velocity growth rates 0.093 for the main DNS
   case and 0.105 for its scalar case, 0.100 to 0.110 for five measured jets and 0.118 for
   Hussain and Clark 1977; scalar growth 0.158 DNS, 0.115 to 0.167 measured; K1u / K1s 0.66
   for the DNS scalar case and 0.66 to 0.87 measured), section 4.2 (the half-width's definition on the excess over the local coflow,
   the entrainment 0.035, the coflow's effect). Fetched 2026-10-29,
   https://escholarship.org/content/qt191048kj/qt191048kj_noSplash_821a0fa2720a5ce7c1642fed646d7ad9.pdf,
   SHA-256 773cec2b...47552fd, 49 pages (scanned, OCR text).
3. Chen, Y.-S. and Kim, S.-W. (1987), "Computation of turbulent flows using an extended
   k-epsilon turbulence closure model", NASA CR-179204; the introduction (the plane jet
   predicted correctly, the round jet 30% over), the submerged jets (experimental 0.11 for
   the plane jet and 0.098 for the round; standard k-epsilon 0.11 with the simplified
   production term and 0.117 with the full one for the plane jet; the extended model 0.111
   plane and 0.108 round) and the coflowing plane jet of Bradbury
   (1965), J. Fluid Mech. 23, 31-64. Fetched 2026-10-29,
   https://ntrs.nasa.gov/api/citations/19880002587/downloads/19880002587.pdf, SHA-256
   a6703ecb...8231aa2, 30 pages (scanned, OCR text).
4. `docs/reports/phase4_scenarios_report.md`: "The decision" (Alex, 2026-10-29), tables S1,
   B1 to B3, G1, M1, Cv1 and Cv2, "The carried limits".
5. `docs/ADR/ADR-014-scenarios-and-time-integration.md`, sections C (the sensor rule, the
   recovery field, what Phase 5 reads) and H (the carried limits); `docs/ADR/ADR-012-turbulence-
   model.md`, sections C (the inlet convention) and D (the design range of supply nu_t), and
   its source 7 (Tominaga and Stathopoulos 2007, the Schmidt number's range).
6. Permana, I., Hidayatulloh, Z. A., Agharid, A. P. and Vishwakarma, P. (2026), "Analysis of
   airflow distribution in Class 100 and Class 1000 semiconductor cleanrooms with a shared
   return air chamber", Energy Engineering 123(11); CFD (RNG k-epsilon) validated on measured
   velocities: turbulence intensity 0% to 2% in the occupied zone, 3% to 6% near the returns.
   Read 2026-10-29 at https://www.techscience.com/energy/v123n11/68931/html.
7. "Dispersion of gas pollutant in a fan-filter-unit (FFU) cleanroom", Building and
   Environment (2007), https://www.sciencedirect.com/science/article/abs/pii/S0360132306000692:
   the page was not readable (HTTP 403); the search result's snippet reports turbulence
   intensity under 2% for about 0.5 m below the unit. Cited for that snippet only.
8. "Measurements of air flow characteristics in a full-scale clean room", Building and
   Environment 31(2), 1996, https://www.sciencedirect.com/science/article/abs/pii/0360132395000399:
   three velocity components by a three-dimensional ultrasonic anemometer, turbulence intensity
   among the quantities presented; the page was not readable (HTTP 403) and its values are not
   used.
9. ISO 14644-1:2015, Cleanrooms and associated controlled environments, Part 1: the class
   limit formula C_n = 10^N (0.1 / D)^2.08 with D in micrometres, rounded to three significant
   figures, and the single-sample volume and time; cited from the standard as the product
   file's thresholds already are (not fetched in this pass).
10. GitHub issue 61 (closed): the pressure outlet's standing imbalance under entering air, the
    reason the product and the validation cases have no pressure outlet.
11. Warhaft, Z. (1984), "The interference of thermal fields from line sources in grid
    turbulence", J. Fluid Mech. 144, 363-387 (DOI 10.1017/S0022112084001646). The primary
    measurement; Cambridge Core gave the abstract only (read 2026-10-29), whose one number is
    a wire-spacing range; the paper's parameters are taken from [13] and its figures are to
    be digitised at step 0.
12. Stapountzis, H., Sawford, B. L., Hunt, J. C. R. and Britter, R. E. (1986), "Structure of
    the temperature field downwind of a line source in grid turbulence", J. Fluid Mech. 165,
    401-424. The companion measurement; not readable in this pass (the ADS and Cambridge
    pages refused the fetch), named as the cross-check.
13. Viswanathan, S. and Pope, S. B. (2008), "Turbulent dispersion from line sources in grid
    turbulence", Phys. Fluids 20, 101514; table I (Warhaft's parameters: sigma_o 1.27e-4 and
    2.5e-5 m, M 2.54e-2 m, x_o / M 20, 52 and 60, U 7 m/s, sigma_u 2.44 and sigma_v, sigma_w
    2.27 m/s at one mesh length, m 1.4, kappa 2.1e-5 m^2/s), equations 3 to 5 (the decay
    law), 10 (the Langevin model, C_0 2.1), 11 to 15 (the plume width), figure 4 and its
    text (the theoretical width against the model's). Fetched 2026-10-29,
    https://tcg.mae.cornell.edu/pubs/Viswanathan_P_PF_08.pdf, SHA-256 e294a5a0...0eb862,
    25 pages.
14. Mohamed, M. S. and LaRue, J. C. (1990), "The decay power law in grid-generated
    turbulence", J. Fluid Mech. 219, 195-214: the measured exponents 1.16 to 1.37 across the
    studies it reviews and 1.24 to 1.33 in its own analysis, as the search of 2026-10-29
    reported them (the paper itself was not fetched); the standard model's 1 / (C_e2 - 1) =
    1.087 with C_e2 1.92 sits below that range.
15. Wilson, J. D. and Sawford, B. L. (1996), "Review of Lagrangian stochastic models for
    trajectories in the turbulent atmosphere", Boundary-Layer Meteorology 78, 191-210,
    https://www.eas.ualberta.ca/jdwilson/jdw_bls_blm96.pdf: the statement that Anand and
    Pope's (1983) model agrees with the Warhaft and Stapountzis data in the far field with
    C_0 = 2.1. Fetched 2026-10-29, SHA-256 57efe94d...4954eaaeb, 20 pages (scanned, OCR
    text); its page 196: "For the data of Warhaft (1984) and Stapountzis et al. (1986), Anand
    and Pope find good agreement in the far field using Co = 2.1, while for the data of
    Sawford and Tivendale (1992), Sawford and Borgas (1994) find Co = 3 gives the best fit."
    Anand and Pope (1983), "Diffusion behind a line source in grid turbulence", Turbulent
    Shear Flows 4, Springer, 46-52, is the theory's source.
