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
*Picture.* The plane jet is the free shear flow whose spreading the standard k-epsilon model
is known to get right: measured half-width growth rates dy_1/2/dx of 0.100 to 0.110 [1, 2, 3],
the model's 0.103 to 0.117 depending on how it is marched and which production term it carries
[1, 3], against the round jet it overpredicts by 30% [3]. The row therefore cannot show the
model spreading a plume too little in general; what it can show is whether this solver, with
its deferred-correction QUICK momentum, its upwind k and eps, its wall functions and its
fixed-flow outlets, reproduces what the model is known to give, and what its scalar does
beside its velocity at Sc_t 0.7. The probe (T1) ran the jet on coarse grids: a 0.1 m slot at
10 m/s into a coflow between walls, the whole far edge a fixed-flow outlet stating no
velocity, so no face can draw air in (#61). The spreading rate is the slope of the half-width
over 15 to 45 slot widths, the velocity's on the excess over the local coflow, the scalar's on
a passive class the jet carries. The first version (30 slot widths high, a 0.05 U_j coflow)
was a confined jet: a plane jet's volume flux grows as the square root of distance and by the
outlet the jet carried more than the coflow supplied, so the walls carried a return flow, the
eddy viscosity filled the channel and the scalar homogenised to the flux-weighted mean (T1's
first rows; the velocity's slope still read 0.115 on 120x60). The second version is 50 slot
widths high with a 0.1 U_j coflow, which supplies about twice what the jet entrains by the
fit range's end. On it the solver converges by the rule on both grids and reads K1 0.093 on
120x100 and 0.086 on 240x200, 14% and 20% under the model's 0.108: prediction (a) holds on
the coarse grid and fails on the row grid as run. A third arm, 0.05 U_j in a 100 slot-width
domain on the coarse resolution (120x200), reads 0.106, 2% under 0.108 and inside the
measured range, so on that resolution the coflow carries most of the gap (the DNS read its
0.09 coflow the same way [2]); a fourth arm, k and eps by UMIST in place of upwind, reads
0.097 at the iteration cap, 4% above upwind, so the scheme's share is about a fifth of the
gap and the grid's the rest.
The scalar spreads 1.22 to 1.29 times faster than the velocity on every converged arm, inside
the measured 1.15 to 1.52 and near the 1.23 a constant eddy viscosity at Sc_t 0.7 gives.
*Options, ranked.*
(1) A plane jet in a weak coflow that supplies its entrainment, as the probe's second version
ran it: the coflow fills the domain so the air the jet entrains comes from the inflow, not
from a recirculation along the walls, and every boundary is one the solver has validated or
runs the product on (velocity inlets, walls, a fixed-flow outlet). The coflow lowers the
growth rate a little (the DNS at a velocity ratio of 11 read about 10% under the still-air
measurements and put part of that on its coflow [2]), and the probe measured it: 0.1 U_j over
50 slot widths reads 0.093 and 0.05 U_j over 100 reads 0.106 on the same resolution (T1). The
row is therefore the 0.05 U_j case in the 100 slot-width domain, whose coflow supplies 4.95
slot flows against the 2.5 entrained (section A's arithmetic). The literature value is
Bradbury's coflowing jet beside the still-air ones [1, section 3; 3].
(2) A jet into still air in a closed box (walls all round but the inlet and the far outlet):
the entrained air returns along the walls as a recirculation, a confined jet, and the
half-width in the fit range feels the return flow; the literature's value is for an unconfined
jet.
(3) A jet with the lateral boundaries open as pressure outlets, the textbook free jet: air
enters through them, which is the treatment issue 61 found cannot close an outlet cell under
this solver and the reason the product has no pressure outlet.
*Recommendation.* (1), at 0.05 U_j in the 100 slot-width domain. The row is `val024_240x400`
in the harness (2.5 cm cells, four across the slot, 96,000 cells; not yet run, step 0's first
task, under an hour on an unloaded core by the 240x200 solve's 6,000 outer iterations in 19
minutes), with `val024_120x200` (the probe's third arm, run) reported beside it as the grid
sensitivity, a case file `configs/validation_plane_jet.yaml`, a loader
`validation.cases.load_plane_jet` and readers in `validation/plane_jet.py` (the half-width
and the fit, pinned by `tests/test_plane_jet.py` on a profile of known half-width). Threshold
proposed, OPEN for Alex: the velocity's K1 within 15% of 0.108, the standard model's value
from the steady similarity solution [1] (0.092 to 0.124, a band that holds the measured range
and the full-production value 0.117 [3]); the scalar's K1 over the velocity's reported against
the measured 1.15 to 1.52 [2, table 7] and not scored until Alex decides whether the scalar's
spreading is a criterion. What the probe leaves open for the row: at 0.1 U_j the slope fell
from 0.093 to 0.086 under refinement, so the 240x400 run may read under the coarse 0.106 by a
similar share, and the band's lower edge (0.092) is where that would be decided; if the fine
row fails the band, the finding is the solver's and the sweep (B) carries it as the stated
spreading uncertainty rather than the row being loosened. The probe's results are in the
probe section below.

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
*Options, ranked.*
(1) The sweep on 160x60 for every arm and every event scenario (the breach, the door, the dust
release), reading per sensor and class the peak over threshold and the first time at or above,
and per sensor row the plume's width at the threshold level and at ten times it (T4's
readings), with the two extreme arms repeated on 320x120 for the breach to show the change of
width between arms is the same on both grids to within the coarse grid's cell. The reason for
160x60: the question is comparative (does the width change between arms, and does any sensor
come within reach of its threshold), the room as it is stops by the rule on 160x60 (ECR-002
step 8), every sensor and opening sits on faces of both grids, and a run costs a fraction of
the fine grid's (T2's flow wall and the march's wall, section B). The cost of the five arms on
the breach is the probe's; three scenarios at five arms is 15 runs, concurrent five at a time.
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

**The case.** A plane turbulent jet in a weak coflow. The literature [1, 2, 3], from the
sources fetched for this pass (section "Sources"), fixes the quantity and its values:

- The half-width y_1/2(x) is the distance from the centreline to where the mean streamwise
  velocity excess over the coflow is half the centreline excess; in the self-similar region it
  grows linearly, y_1/2 = K1 (x - x0), and K1 is the spreading rate. Measured values for plane
  jets into still air: 0.100 (Gutmark and Wygnanski 1976, Re 30,000), 0.104 (Browne et al.
  1983, Re 7,620), 0.110 (Thomas and Chu 1989; Thomas and Prakash 1991; Ramaprian and
  Chandrasekhara 1985), a range of 0.100 to 0.110 [1, table 1; 2, table 2]. Bradbury's jet in
  a coflow of 0.16 U_j is self-preserving beyond about thirty slot widths [3, section 3; 1
  cites 0.109 for it through its review]. Chen and Kim quote the experimental plane jet value
  as 0.11 [3].
- The standard k-epsilon model: 0.108 from the steady similarity solution tabulated by Wilcox,
  0.103 from an unsteady RANS computation (Magi et al.), both as Premnath and Abraham's table
  1 collects them [1]; Chen and Kim's parabolic march gives 0.110 with the boundary-layer
  production term mu_t (dU/dy)^2 and 0.117 with the full production term mu_t S^2 [3, section
  2 of the results]. This solver carries the full term (ADR-012 A: P_k = mu_t S^2, S^2 = 2 S_ij
  S_ij), so 0.108 to 0.117 is the model's expected range here, and the measured 0.100 to 0.110
  lies inside the 15% band about 0.108.
- The scalar's spreading beside the velocity's: the sources give it for the temperature in
  moderately heated plane jets. Stanley and Sarkar's table 7 [2] lists the measured velocity-to-
  scalar growth ratios K1u / K1s of 0.66 (Ramaprian and Chandrasekhara 1985), 0.81 (Browne et
  al. 1983), 0.71 (Jenkins and Goldschmidt 1973) and 0.87 (Davies et al. 1975), so the scalar
  half-width grows 1.15 to 1.52 times faster than the velocity's; their DNS gives 0.66 (1.51)
  and reads it as consistent with Sc_t about 0.7. Under a constant eddy viscosity across the
  jet the scalar profile is the velocity profile to the power Sc_t, which at 0.7 puts the
  scalar's half-width 1.23 times the velocity's (the sech^2 profile's half points at 1.08 and
  0.88 of its width scale); the model's ratio is expected near that, inside the measured range.

**The boundaries this solver can run it with.** A free jet entrains: the air it draws in must
come from somewhere, and the textbook free jet takes it through open lateral boundaries, the
pressure outlets whose inflow issue 61 found this solver's zero-gradient copy cannot close.
The design avoids it as the probe's second and third versions did
(`docs/reports/probe70/jet70.py`): the left edge is three velocity inlets, the slot at U_j
between two coflow inlets, the top and bottom are no-slip walls under the wall functions, and
the whole right edge is a fixed-flow outlet that states no velocity, so it
takes the inflow's remainder at one face velocity and the domain has no pressure outlet
(ADR-014 J's closed-box rule, refused at load otherwise). The coflow has to feed the
entrainment, and the probe's first version showed what happens when it cannot. From the
self-similar constants of [2] (the centreline excess decaying as 2.28 U_j sqrt(b / x), the
half-width growing at 0.1 x, the excess profile integrating to 2.16 half-widths), the jet's
excess volume flux is about 0.49 U_j b sqrt(x / b): 2.3 slot flows entrained by 45 slot
widths and 2.5 by the outlet at 60. The first version's coflow, 0.05 U_j over 29 slot widths,
supplied 1.45 slot flows, so the walls carried the difference back toward the inlet: reverse
flow of about 1 m/s at the walls beyond 20 slot widths, an eddy viscosity of 0.1 m^2/s across
the whole channel and the scalar at the flux-weighted mean 0.41 everywhere past 20 slot
widths (`results/builder70/jet_120x60.npz`), while the flow cycled to the iteration cap. The
second version's coflow, 0.1 U_j over 49 slot widths, supplies 4.9 slot flows, twice the
entrainment, and the solve converges by the rule; the third version's, 0.05 U_j over 99 slot
widths (walls 50 slot widths from the centreline), supplies 4.95 and converges too, and is
the row's domain since the coflow's own effect on the slope is the larger of the two (T1,
decision A). The outlet's uniform exit profile is 15 slot widths beyond the fit range's end,
where the half-width is about five slot widths; its reach upstream is visible in
`docs/reports/probe70/jet70.png` from about 50 slot widths, outside the fit range.

**The readings** (jet70.py's docstring): the velocity's excess over the local coflow (the
column's mean outside 1.0 m of the centreline, the four wall cells excluded), the half-width
from the two half-excess points by linear interpolation, the least-squares line over 15 to 45
slot widths; the scalar's half-width the same way on the passive class marched to a steady
field on the converged faces and nu_t, the jet carrying C = 1 and the coflow clean.

**The grids.** 240x400 (2.5 cm cells, four across the slot; the row, to be run at step 0) and
120x200 (two across; the sensitivity, the probe's third arm). The slot's resolution sets the
virtual origin, not the self-similar slope [2, section 4.2 on the virtual origins' scatter],
so the row scores the slope; but T1's 0.1 U_j pair shows the slope itself falling 7% from
two to four cells across the slot, so the row's grid pair is reported together and the
threshold is read on the finer.

**The row.** `configs/validation_plane_jet.yaml` carries the probe's case with the product's
model and solver keys (ADR-012 D's note of 2026-10-09: the standard variant, upwind k and
eps, ten momentum sweeps, `pressure_rtol` 1e-4, the error-estimate stop with
`mass_imbalance_tol` for the grid's smallest cell). `validation/plane_jet.py` holds
`half_widths(mesh, field, background)` and `spreading_rate(xc, widths, fit_range)` and the
constants (the slot width, the fit range, the literature values with their sources);
`validation.cases.load_plane_jet` refuses a grid that puts a slot edge off a face.
`tests/test_plane_jet.py` pins the readers on a profile of known half-width (a sech^2 whose
half point is exact) and the loader's refusal; the harness rows `val024_240x120` and
`val024_120x60` run the case, march the scalar and report K1 for both fields, the virtual
origin and the fit's residual, scored by the threshold Alex sets. The row's cost is T1's.

**Threshold, OPEN.** Proposed: K1 of the velocity within 15% of 0.108 on 240x400; the scalar's
ratio reported against 1.15 to 1.52. An alternative, stricter: within 10% of the measured
0.110 (0.099 to 0.121). The probe's values are in the probe section; the threshold is proposed
before them as the predictions were.

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

**What is read.** Per arm, per scenario, per class: each sensor's peak over threshold and its
first time at or above (T3's readings); on each sensor's row, the plume's width at the
threshold level and at ten times it, at the end of the event and the largest over the frames,
and the extent of the threshold cells (T4's); the arrival map's fraction of cells reached and
its earliest arrival (T5's). The plume's width is the quantity the inputs move; the peak over
threshold is the margin a sensor keeps; the map says whether any position would have read it.

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
recovery tracker's are, so a restart continues it bitwise. Written at the end of the run and at
each checkpoint flush as `/arrival/` (`ArrivalRecord`: thresholds, onsets with their event
names, `first_above` [classes, ny, nx], t_final; `write_arrival` and `read_arrival` on
`RunFile`). The clean-out gets the map too; its arrival is t = 0 everywhere above threshold,
which is what the clearing-time companion needs to say nothing new.

**Run files without it.** `placement.arrival_from_frames(run_file)` reads the stored frames
and returns the map at the output cadence's resolution, flagged as such. The Phase 4 run files
and probe B's are read this way (T5), and section F's sweep can be read before step 3 is
built.

**Latency.** The onset of an event that carries contamination is its `start` (a `replace`
with a concentration key, a `source`, or the initial condition at t = 0). For each such event
e, the latency at a cell is `first_above` minus `start_e` where `first_above` lies in
[start_e, start of the next such event or t_end); a cell whose first arrival is before the
event belongs to the earlier one. With one such event per reference scenario the rule reduces
to arrival minus 10 s (the door, the breach, the dust) and zero for the clean-out. A cell never
reached has infinite latency and the report says "never" with the run's end beside it.

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
- the worst-case rule at any count: exact by a threshold sweep. For each distinct latency T in
  increasing order, the pairs a cell covers within T is a fifteen-bit mask; the layouts of the
  count whose masks OR to all ones exist or not, which is settled over the distinct masks (a
  few hundred) rather than the cells, and the smallest T with a cover is the optimum, the cover
  its layout (ties broken by the mean);
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
| 3: the spreading check (A and B) | Steps 0 and 1 of H, before any detection result | A: T1's two solves and marches on a workstation core each (the probe section gives the walls); B: 15 coarse runs concurrent five and two fine runs (section B) | Whether the plumes' widths and the sensors' margins move with the assumed inputs: if a sensor comes within reach of its threshold at an arm, the report states detection results as a range over the arms; if none does, the "never" is geometric and the maps carry the placement question |
| 2: a third clean-out grid | Step 1, beside the sweep (a run, no code) | One 160x60 clean-out, 240 s at Courant 0.4: about an eighth of the product grid's 2.6 hours (table M1), under an hour with its flow solve | Whether the mean's 100:1 ratio and the local recovery field can be quoted as numbers of the room: a 160x60 value between 80x30's 3.0 and 320x120's 4.8 shows a series not yet converged and the numbers stay comparative; the room recovery time (25.6 and 27.9 s) is unaffected either way |
| 1: the budget residual's mechanism | Before Phase 6's finer grids, after the Phase 5 gate, a prompt of its own | One session: a sealed-box march (VAL-014's) with a source and an interval switch at a known time on 40x15, the budget read every step, the residual's accrual against the content and its sign at the switches (the records put the sign changes near 10.5 and 25 s in the dust release and 80.4 s in the breach, and B3 rules the flow's continuity out by two orders) | The budget instrument (REQ-T05's verification) if the residual is bookkeeping between the content and the running sums or the source's integral at a switch; the scheme's inlet booking if the limited face value at an inlet face differs from the inflow the budget books. Neither moves a Phase 5 latency: the residual is 1e-6 of the supply against thresholds at 1e-2 of the inlet value |

## G. Requirements review, REQ-A01 to A06

| Requirement | What it asks | Testable as written | Proposed rewording (where needed) |
|---|---|---|---|
| REQ-A01 "detect threshold exceedance within one timestep of occurrence" | A latency bound on the instrument | No: it holds by construction when the monitor reads every step, so no test can fail it; "occurrence" is the step's reading | "The alert monitor shall read every sensor after every transport step and raise an alert at the first step whose reading is at or above the class's threshold, so that it adds no delay of its own to the detection latency, whose resolution is the step." Verified by VAL-009 as section C defines it |
| REQ-A02 sensor locations configurable in YAML | The sensors section | Yes (unit: a sensor added or moved in the mapping changes the cell read; a SOLID cell refused; the face rule) | Add the reading rule: "a sensor reads the cell containing it, the one below or to the left of a position on a face (ADR-014 C)" |
| REQ-A03 thresholds per class following ISO 14644-1 | The thresholds mapping | Partly: configurable per class is testable; "following ISO 14644-1" is not without the formula | "Contamination thresholds shall be configurable per particle size class, either as a mapping by size or as an ISO 14644-1 class number from which the loader computes the limits by the standard's formula rounded to three significant figures; the product file's Class 5 values are the test" |
| REQ-A04 latency per sensor per scenario from event onset | The report's core number | Yes once onset is defined (integration test on a coarse run) | Append: "the onset is the start of the event that carries the contamination (a replacement with a concentration, a source, or the initial condition at t = 0); a sensor never at or above threshold within the run reports no latency and the run's end" |
| REQ-A05 multiple configurations across scenarios to identify optimal placement | The placement question | No: "optimal" names no rule | "The monitor's placement module shall score any set of sensor layouts over a set of scenario run files by stated rules (worst case, mean, coverage within a stated time), rank them, and search the stated candidate positions for the layout of a stated count that is best under a rule, exact for one and two sensors and for the worst-case rule at any count, reproducible from the run files alone" |
| REQ-A06 reads but never modifies | Separation of concerns | Yes, and by test as well as review: the fields' bytes before and after `update`, the writeable flag False | Append: "verified by test: the fields are bitwise unchanged across every monitor call" |

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
| 0 | 71 | The coverage debt: tests for the 66 uncovered statements of table Cv2 of the gate report (47 `raise` lines, 19 fallbacks and early returns), each exercised by its own test or recorded as unreachable with the reason; VAL-024's case file, loader, readers and harness rows | `tests/test_plane_jet.py`; the new rejection-path tests across `test_config.py`, `test_io_manager.py`, `test_solver_transport.py`, `test_turbulence.py`, `test_mesh.py`, `test_time_integration.py`, `test_forces.py`, `test_particles.py`, `test_staggered.py`, `test_solver_staggered.py` | VAL-024, scored by Alex's threshold |
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

## Probes (committed before the runs; results after)

**Predictions (orchestrator, 2026-10-29).**
- (a) The code's standard k-epsilon plane-jet spreading rate falls within 15% of the
  literature's value for standard k-epsilon.
- (b) On 160x60, the filter breach at the top of the supply-intensity range still leaves every
  sensor below its threshold, with the plume's width at sensor height grown by under a factor
  of two.

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

*Note (2026-10-29, after the first jet version ran, before the second).* The first version's
domain (30 slot widths high, a 0.05 U_j coflow) confined the jet (section A: the coflow
supplied 1.45 slot flows against 2.5 entrained), so the domain was revised to 50 slot widths
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
iterations by the rule. The scheme's share is smaller: k and eps advected by UMIST in place
of upwind on the 0.1 U_j arm reads 0.0973, 4% above the upwind arm's 0.0932, at the
iteration cap rather than by the rule (the limited scheme on k and eps cycles, as ADR-012 C
found on the room). So of the 20% on the row grid, the coflow accounts for about two thirds
on the coarse resolution, the k and eps scheme for about a fifth, and the remainder is the
grid's (the 7% fall from two to four cells across the slot). The confined first version
read 0.115 (two cells across the slot) and 0.125 (four) at the iteration cap, both with the
return flow, and its scalar never settled; those rows are evidence for the domain, not
measurements of the model.

*The scalar beside the velocity.* 1.224, 1.243 and 1.288 on the three converged arms (the
scalar's K1 0.114, 0.107 and 0.136), inside the measured 1.15 to 1.52 and within 5% of the
constant-eddy-viscosity estimate 1.23; the scalar marches reached a steady field in 6 to 15
simulated seconds.

*Prediction (b): HOLDS.* On 160x60 the breach at 10% intensity (0.1 m) leaves every sensor
below its threshold in every class, the nearest reading 1.2e-11 of threshold at above_gap_1
(T3), and the plume's width on the sensors' row at 2.5 m is 1.10 m against 1.05 m at the
file's 5% (T4), a factor 1.05. The widest arm of the range, 10% at 0.3 m (supply nu_t 99 nu,
six times the file's), reads 1.20 m (a factor 1.14) and brings above_gap_1 to 7.1e-7 of
threshold; the narrowest, 2% at 0.05 m, reads 0.95 m; Sc_t 0.2 on the file's flow reads
1.10 m and 2.5e-9 of threshold. The width at ten times the threshold moves by one cell or
none. Every flow state stopped by the rule in 1,953 to 3,067 outer iterations (T2); a march
of about 15,000 steps took 610 to 700 s wall with five processes sharing the machine.

*The builder's (a) and (b).* (a) was wrong on the row grid in the direction it did not expect
(the slope under the band, not the virtual origin moved); its ratio expectation of about 1.2
held. (b) held in every part, including the geometry: the arrival maps (T5, `arrival70.png`)
put the earliest arrival 0.24 to 0.47 s after the breach's start in the cell under the panel's
edge at 2.8 m height, the reached cells (17% to 20% of the room's free cells) under the panel,
on the litho tool's top and in both gaps to the floor returns, and the file's four sensors
outside the reached set at every arm.

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
- The plane jet becomes the solver's first free-shear validation row and the spreading
  limit is bounded rather than open; VAL-021 stays closed.

## Alternatives considered

- A monitor that samples a volume or integrates over a counter's sample time (C's options 2
  and 3): deferred, the sample time to be stated beside the latency.
- The map from the frames only (D's option 2) and the search by rerunning (D's option 3):
  rejected for the output cadence's resolution and the cost.
- A still-air jet in a closed box or with open lateral boundaries (A's options 2 and 3):
  rejected for the confinement and for issue 61.
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
   450, 2002, with Mellado); tables 2 and 7 (velocity growth rates 0.093 DNS, 0.100 to 0.118
   measured; scalar growth 0.158 DNS, 0.115 to 0.167 measured; K1u / K1s 0.66 to 0.87
   measured), section 4.2 (the half-width's definition on the excess over the local coflow,
   the entrainment 0.035, the coflow's effect). Fetched 2026-10-29,
   https://escholarship.org/content/qt191048kj/qt191048kj_noSplash_821a0fa2720a5ce7c1642fed646d7ad9.pdf,
   SHA-256 773cec2b...47552fd, 49 pages (scanned, OCR text).
3. Chen, Y.-S. and Kim, S.-W. (1987), "Computation of turbulent flows using an extended
   k-epsilon turbulence closure model", NASA CR-179204; the introduction (the plane jet
   predicted correctly, the round jet 30% over), the submerged jets (experimental 0.11 and
   0.098; standard k-epsilon 0.11 with the simplified production term and 0.117 with the
   full one; the extended model 0.111 and 0.108) and the coflowing plane jet of Bradbury
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
    reason the product and the jet case have no pressure outlet.
