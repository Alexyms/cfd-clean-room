# ADR-014: Scenarios and Time Integration: Piecewise-Steady Flow, Time-Accurate Transport, and a Transient Solve for the Cylinder

## Status
Accepted 2026-10-16 (Alex): the recommendation of each of the twelve decisions below, as
written after rounds 1 and 2 of prompt 59, with one amendment of the same date, the equipment
vents (section J, built as step 0b of section G; its limit in section H). The dated note under
"Decisions for Alex" records it. The build has begun with step 0 (prompt 60).

Written 2026-10-16 (prompt 59, branch `phase4/design`), before the Phase 4 build, from the
solver as ADR-010 to ADR-013 record it, the Phase 3 gate report's stated limits
(`docs/reports/phase3_transport_report.md`) and the probes under `docs/reports/probe59/`, whose
records are in `results/builder59/` and whose tables are `docs/reports/probe59/tables59.md`.
Three decisions were taken by Alex on 2026-10-16 and are recorded below as given; the rest were
put to Alex as a picture first, then ranked options with their consequences, then the section
carrying the detail, as ADR-011 and ADR-013 did. The document gains a planned-against-built
table at the Phase 4 gate, as ADR-011 and ADR-012 did. Bracketed numbers point at the sources
at the end; "T1" to "T7" are the tables of `tables59.md`, generated from the records by
`tables59.py`, whose `--check` regenerates and compares them.

## Decisions already taken (Alex, 2026-10-16), recorded as given

1. **Two tiers of time integration.**
   - *Tier 1, the room: piecewise-steady flow, time-accurate transport.* When an event changes
     the flow's boundaries, the flow is re-solved to its new steady state, warm-started from the
     last, and the particles are marched through time on it. The room's air turnover is short
     against particle residence (minutes) and against most events. (Alex's note put the
     turnover as the room length over the supply speed, about 8 m / 0.45 m/s, roughly 20 s; T6
     gives that ratio as about 18 s and the free area over the supply flow as about 5.6 s, the
     substitution made here.) Events that
     change only the transport (a filter breach: contaminated supply air; an equipment dust
     release: a source) need no flow re-solve.
   - *Tier 2, where unsteadiness is the point: time-accurate flow.* The square cylinder (Phase
     4's validation and showcase, laminar vortex shedding; Sohankar, Norberg and Davidson 1998
     [1]) needs a true transient: an implicit time step with outer iterations inside it.
     Turbulent rooms are not run time-accurately in 2D; resolved eddies belong to Phase 6's 3D
     work.
2. **Short events are modelled as their steady extreme for their duration.** A door opening for
   a few seconds is represented by the fully open state's steady flow for the opening's
   duration, a worst case, not by the door's motion. No onset ramp is needed under tier 1;
   section A says what the plan's "ramp event onset" risk becomes.
3. **A fourth reference scenario, the clean-out (recovery).** The room starts with contaminated
   air, uniform (after a shutdown, say), the supply clean, and the question is how long until it
   is back within its specification. Section C defines "back within spec".

## Decisions for Alex
Each decision is a picture first, then the options ranked with their consequence, then the
section that carries the detail. The ranking is the builder's, from the probes and the records.

**Accepted (Alex, 2026-10-16).** Each decision's recommendation is taken as written, after the
two rounds of prompt 59 (the proposal, and prompt 59b's answer to review 59): decision 1 the
fingerprint cache and the warm start, 2 Courant 0.4 for the scenarios with Heun's method named
for Phase 6, 3 BDF2, 4 SIMPLE outer iterations per step, 5 the velocity step below 1e-4 U, 6
the two-way door exchange, 7 the `scenario` section with `base:`, 8 every sensor below
threshold with the 100:1 time beside it, 9 one HDF5 run file, 10 VAL-008 on the wide front, 11
interior clustering with the freestream lateral boundaries, 12 the plan's three modules and the
build order of section G. Decision 11's recommendation leaves its thresholds OPEN, and they stay
so until the tier-2 step (step 6) is prompted. One amendment of the same date: the equipment
vents, airflow only (section J), built as step 0b, before the scenario schema, so the schema and
the reference scenarios carry vents from the start; the heat a cooling discharge carries is a
limit (section H).

**1. What triggers a flow re-solve, and the warm start (section A).**
*Picture.* Under tier 1 the flow is a sequence of steady states, one per distinct set of velocity
boundaries, and the particles see a step change in their faces at each switch. The question is
which events change the flow, how the solver starts from the last state instead of from rest,
and what that costs. The probe (T1) solved the product room to the open-door state from rest and
from the converged closed-door state, and back. The warm start saved about a sixth of the
outer iterations, not the two thirds prediction (a) expected: the `error_estimate` stop comes
where the slow mode's geometric decay puts the estimated iteration error below its tolerance.
The warm arm's first residual is about a third of the rest arm's, but what the start saves is
set by where the slow mode starts: the two residual tails are parallel and about 700 outer
iterations apart at about 500 per decade (`warm.json`, `rest.json`), so the warm state begins
the slow mode about 20 times smaller than rest does, and those 700 iterations are the saving.
The cost of a flow state is therefore about one steady solve whatever it starts from, and what
tier 1 can save is solving a state twice.
*Options, ranked.*
(1) The loader decides at load time: an event changes the flow if, and only if, it changes a
key the velocity layer reads (a segment's type, extent or velocity). The run's distinct flow
states are enumerated before anything is solved, each is solved once per run and cached by its
boundary fingerprint, and each solve is warm-started from the nearest state already solved
(`solve_steady(initial=...)`, the faces, pressure, k and eps of a kept state, which the probe
supplied by patching the two allocations and the design adds as an argument). A state's cost is
T1's warm arm, about 0.85 of a solve from rest; a return to a state already solved costs
nothing, which for the door (closed, open, closed) makes the run's flow cost two solves of the
three (T1: 0.71 of solving all three), or one solve when the closed state is already held in a
run file (section D).
(2) Re-solve at every event boundary, from the last state, no cache: the door's closing costs a
second solve (T1's back arm) every time it closes.
(3) Re-solve from rest at every switch: T1's rest arm per event.
*Recommendation.* (1). The consequence is a `FlowState` type (faces, p, k, eps, nu_t, the stop
record) that `solver_staggered.py` takes and returns, which section D's checkpoint stores as it
stands; and a note for the deferred efficiency pass (PROJECT_PLAN, "Deferred"): the re-solve's
cost is the slow mode's, and an outer iteration that adapts its under-relaxation to the damping
it observes is the change that would shorten it, not a better start.

**2. The transport step for the scenarios (section A).**
*Picture.* The transport march, not the flow, is where a scenario's time goes. On the product
grid at the file's Courant number of 0.1 the explicit scheme takes about 1,100 steps per
simulated second per class, and the five classes together cost about 93 wall seconds per
simulated second on this machine (T6): a 60 s scenario takes about an hour and a half of
marching, a 600 s clean-out about 15 hours. The step is set by the fastest cell, a floor-return
cell whose two directional rates sum to about 110 per second (1.5 m/s along the floor and
1.2 m/s into the return over 2.5 cm; T6's step at 0.1), not by the room. The Courant number is an accuracy choice
below the limited scheme's stability bound of 1/2 (ADR-011 C): the forward Euler error
steepens a pulse in proportion to it (ADR-011 H's table: peak and L2 at 0.1, 0.25 and 0.4).
The steady comparison ECR-002 step 8's marches made between 0.4 and 0.1
(`docs/reports/ecr002_step6_comparative.md`, section 5.4) cannot see that error: the explicit
step's steady state solves the same discrete equation at any step, so two marches that agree at
the end agree on the equation, not on the path. The scenarios are scored on transient
quantities, a recovery time or a detection latency, which is where VAL-004's shape error at 0.4
(L2 38% against 11% at 0.1 for the oblique pulse, ADR-011 H) would show. The probe of prompt 59b
measured the time error on the quantity itself: the clean-out on 160x60 at 0.1 and at 0.4 (T7),
every sensor's first time below 1e-2 and 1e-3 of the initial value, both classes.
*Options, ranked.*
(1) `cfl_number` 0.4 for the scenarios, 0.1 kept for the validation rows: four times fewer
steps than 0.1, since `stable_dt` is linear in the Courant number (about 23 wall seconds per
simulated second on the product grid from T6; T7's two arms differ by 3.9 in wall time for 4.0
in steps). What it costs in the answer, from T7: the sensors' recovery times at 0.4 are within
3.7% of those at 0.1, earlier at 0.4 in every case, with the largest differences at the three
sensors under the supply that clear within two seconds (2.2% to 3.6%) and the near-door sensor,
which clears last at 29 and 55 s, within 0.03%; the free-area mean's decay is the same curve to
three figures. A recovery or detection time read at 0.4 is therefore a few percent early on the
fastest sensors and unchanged on the slow ones, against a comparative purpose (SYSTEM.md
section 1) that ranks layouts and sensor positions.
(2) Heun's method (the two-stage SSP Runge-Kutta member, ADR-011 C's recorded alternative): the
same bound of 1/2, second-order in time, so 0.4 costs no steepening; two advection evaluations
per step, so about twice option (1)'s cost per simulated second (about 46 wall seconds) and half
of option (3)'s, for the answer at 0.1's accuracy or better. It touches `solve_timestep` only.
ADR-011 deferred it to Phase 6, where the step is paid for on the GPU.
(3) `cfl_number` 0.1 everywhere: the validation rows' accuracy at the full cost of T6, about 93
wall seconds per simulated second, four times option (1).
*Recommendation.* (1) now, with T7's few percent stated beside every recovery and latency
figure the scenarios report, and (2) named as the first Phase 6 transport change, so the
scenarios are repeatable on the GPU at the same Courant number with the time error removed
rather than tolerated. If a scenario's figure must hold to better than T7's difference, (3) is
the fallback at four times the cost. The scenario files state their Courant number; nothing in
the solver changes.

**3. Tier 2's time discretisation (section A).**
*Picture.* A time-accurate step needs the old time level in the momentum equation. Implicit Euler
adds `rho V / dt` to the diagonal and `rho V u_old / dt` to the source, first order in time;
second-order backward differencing (BDF2) keeps two old levels with weights 3/2, -2, 1/2;
Crank-Nicolson, the paper's scheme [1], averages the old and new fluxes. The probe used implicit
Euler because it needs one old level and no change to the deferred-correction source, and the
steady solver's `dt` key already exists for it.
*Options, ranked.*
(1) BDF2 with an implicit Euler first step: second order in time, the same diagonal dominance as
implicit Euler (the transient coefficient is 3/2 of it), two stored levels. The shedding period
is resolved by 137 to 274 steps at the paper's steps (T4), where first order in time damps the
oscillation's amplitude and shifts its period in proportion to dt; the Strouhal threshold of
section F is what pays for that error, and BDF2 removes it at the cost of one more stored field.
(2) Implicit Euler: what the probe measured; a step small enough makes it right, which costs
steps (T4's hours scale with 1 / dt).
(3) Crank-Nicolson as the paper: second order, but half the convective and diffusive fluxes are
explicit, so the deferred-correction and under-relaxation machinery would carry explicit QUICK
fluxes of the old level, which ADR-011 C showed unstable under forward Euler; a scheme to build
and test on its own.
*Recommendation.* (1). The term lives in `momentum.py` as a predictor option (the probe's
subclass, `ImplicitEulerPredictor` in `cylinder59.py`, is the implicit Euler form), the two old
levels in the transient loop of `solver_staggered.py`, nothing in `time_integration.py`.

**4. The pressure-velocity scheme inside a step (section A).**
*Picture.* Inside a time step the pressure and velocity must be made consistent. SIMPLE repeats
the steady outer iteration (predict, correct, under-relax) until the step's criterion holds,
which the probe did with the existing machinery; PISO (Issa 1986 [8]) takes one prediction and
two or more corrections without outer iteration, relying on a small step; the paper used SIMPLEC
with outer iterations [1].
*Options, ranked.*
(1) SIMPLE outer iterations per step, as probed: no new corrector, the under-relaxation and the
ten momentum sweeps of ADR-012 D's note as they stand, the step's criterion of decision 5. From
the start-up field the step converged in 3 to 4 outer iterations to 1e-4 U (T2), and in 7 to 8
under developing shedding (T3), the figure the showcase's cost rests on (T4); the paper's
SIMPLEC needed 23 to 29 per step with its criterion at the Neumann outlet [1, Table II].
(2) PISO: fewer momentum assemblies per step, but a second corrector to build and a step bound by
the scheme's own accuracy rather than by the physics; the existing corrector's `d` coefficient
and projection would be reused twice per step with a modified right-hand side.
*Recommendation.* (1).

**5. The per-step convergence criterion (section A).**
*Picture.* Each time step's outer iterations stop by a rule. The steady rule, `error_estimate`,
is built for a solve that converges geometrically over thousands of iterations and reads its
rate from a window of steps; within a time step there are a handful. The velocity-step rule
(`convergence_tol`, the largest cell-centred change per outer over the reference velocity) is
what the probe used, at three levels.
*Options, ranked.*
(1) The velocity step below 1e-4 U per outer, with the continuity reading recorded every step
(the worst cell imbalance, which T2 lists) and a cap per step that fails the run rather than
hides a stall. Measured 3 to 4 outer iterations per step, every step, from the start-up field
(T2, arms A to C and the counts to 1e-4 in D to F), and 7 to 8 per step under developing
shedding (T3, arm B's long march), which is the per-step cost the showcase is priced on.
(2) 1e-5 U: 7 outer iterations on most steps, but a slow mode stalls some steps at that level
for 34 to 212 outer iterations (T2, arm D's rows; `cylinder_D.json`), which the cap would have
to tolerate; the per-step cost is then set by the stalls, not the typical step.
(3) A normalised residual sum as the paper's (the sum of absolute residuals over the inflow
momentum flux below 1e-3 [1]): a different instrument to build; its count was 23 to 29 per step
in the paper.
(4) The `error_estimate` rule per step with a short window: its conditions (a) to (d) hold by
construction after one corrected iteration on the faces, and its rate estimate needs more
iterations than a step has.
*Recommendation.* (1), with the step's imbalance recorded and the cap at a few hundred. The
stall at 1e-5 is not located here; if the showcase's Strouhal number proves sensitive to the
criterion, the tier-2 step (section G) measures 1e-5 against 1e-4 on one period.

**6. The door (section B).**
*Picture.* The room is positively pressurised in practice: the supply exceeds the ducted returns
by an offset that leaks out through the envelope at 5 to 20 Pa (ISO 14644-4's range as ASHRAE
reports it [4]). The configured room has no offset (the returns take what the hood leaves of the
supply, exactly, ADR-012 decision 1), so a door leak in it carries nothing in or out until the
design says what the door does. With the door fully open the pressure difference collapses, the
offset leaves through the doorway at a low speed, and air is exchanged both ways across the
opening by the door's swing, by people passing and by any temperature difference: measured
exchange volumes for a hinged door in still air are 1.2 to 2.4 m^3 per opening cycle, with a
passage adding about 0.4 m^3 (Kalliomaki et al. 2015 [6]); door operation can cancel and even
reverse the pressurisation across the doorway, and higher differentials recover sooner
(Bhattacharya et al. 2021 [5]).
*Options, ranked.*
(1) Two-way exchange as the steady extreme: the door segment becomes an inlet of corridor air over
the lower part of the doorway at an exchange velocity, carrying a configured corridor
concentration per class, and a fixed-flow outlet over the upper part at the same velocity plus
the offset (zero in the configured room). The returns keep their shares exactly, since the
doorway's net flow is the offset. The exchange velocity and the split height are configuration,
stated as an assumption: the probe used 0.25 m/s over the lower half (`warm59.py`), which a
1.2 to 2.4 m^3 exchange over an assumed 5 s opening of a 0.9 m wide door puts just under
(0.26 m^2/s per metre of depth across 1.05 m against 0.27 to 0.53; the opening's duration is
the probe's assumption, not the source's). Corridor air enters, so the scenario produces contamination
for Phase 5 to detect.
(2) Outflow only, the pure positive-pressure picture: the door a fixed-flow outlet at the offset
velocity, no corridor air; the scenario then shows a clean room staying clean, a control, not an
alert case.
(3) Inflow only, the plan's "door seal leak" read literally: corridor air leaking in through the
seal is the physics of a room at negative pressure or of the pressure reversal during a door
operation [5]; as a steady extreme it overstates both.
*Recommendation.* (1), with (3) kept as the seal-leak variant of the same schema (an inlet over
the seal's extent at a leak velocity), so the plan's name survives as a configuration, not a
different model. The flow re-solve is needed in both: the velocity layer changes.

**7. The scenario schema and configuration inheritance (section B).**
*Picture.* A scenario is a base configuration, an initial condition and a list of events; REQ-C03
asks scenario files to inherit from a base and override only what differs, and today the loader
reads one file whole. Every event has a start, a duration and what it changes: a boundary
segment's replacement (the door: section B's two segments), a transport-only change to a segment
(the filter breach: `hepa_filtered` false and a `concentration`), a source (equipment dust: a
box and a rate per class), or the initial field (the clean-out).
*Options, ranked.*
(1) A `scenario` section in the file with `base:` naming the file it extends; the loader merges
the scenario's mappings onto the base's recursively (lists replace whole), validates the merged
mapping as a file, then validates the events: types and ranges, the end within `t_end`, no two
events on one segment overlapping in time, every source inside non-SOLID cells, the per-class
arrays of the configured length, and every flow state the events produce loaded through the
same validation at load time, so a state that cannot load fails before anything runs (REQ-C02).
The merged mapping is what the run records.
(2) Two files on the command line, base and scenario, merged by the runner: the same rules, the
link recorded outside the file.
(3) Whole scenario files, no inheritance: REQ-C03 unmet and four copies of the room to keep
aligned, which `tests/test_layouts.py` already guards against for the two layouts.
*Recommendation.* (1).

**8. "Within spec" and the recovery outputs (section C).**
*Picture.* The default configuration carries a threshold per class, ISO 14644-1 Class 5 limits
(the `thresholds` section). ISO 14644-3's recovery test measures the time for a challenge at 100
times the target level to fall to the target, the 100:1 recovery time, with an alternative from
the slope of the semi-log decay; it sets no acceptance limit and is meant for non-unidirectional
rooms at rest [2, 3], which the configured room, with a ceiling supply over 7 of its 8 m at
0.45 m/s, is not by the usual reading: the 100:1 number is reported outside the test's intended
scope, as a figure a reader knows, not as the test. EU GMP Annex 1 gives a clean-up period of
15 to 20 minutes as guidance (2008 text), "less than 20 minutes" in the 2023 note [3]. A
well-mixed room decays as C(t) = C0 exp(-N t); for this slice N is the supply flow over the free
area, and the 100:1 time is ln(100) / N, about 26 s (T6). For a displacement flow the well-mixed
line is a bound the room should beat, since the supply sweeps the slice in one turnover; lying
above it is the signature of short-circuiting and of recirculation zones, which T7 measures.
*Options, ranked.*
(1) Every sensor below its class's threshold, for every class, and staying below: the alert
system's own reading (REQ-A01, A03, A04), and ISO 14644-1's rule that each sampling location
meets the class limit, the sensors standing for the locations. Reported beside it: the 100:1
recovery time at each sensor and for the free-area mean, the standard's number, defined when the
initial level is at least 100 times the target (the clean-out starts there by construction).
(2) The free-area mean below threshold: the well-mixed comparison's own quantity, which hides
the dead zones the field of local recovery time is there to show.
(3) The 100:1 drop alone: independent of the thresholds, so a room that starts cleaner than 100
times its limit has no recovery time.
*Recommendation.* (1). Two outputs carry the physics: the free-area mean against the well-mixed
exponential, which shows where the room is not well mixed, and the field of local recovery time
(per cell and class, the last time the cell is above threshold), which shows where contamination
lingers. Phase 5 reads the per-sensor recovery times as the alert's clearing, and the field tells
it where a sensor would have cleared last.

**9. Output, checkpoint and restart (section D).**
*Picture.* Phase 3 wrote concentration frames through `FieldHistory` (npz) and nothing else from
`src/`; every flow record of ECR-002 was written by a probe script, and the measurements that
reached a cap (prompts 49 and 52) could not be continued because no state was kept. h5py is
already a dependency (`requirements.txt`) and the plan names `io_manager.py` for output and
checkpointing.
*Options, ranked.*
(1) One HDF5 run file per run, written by `io_manager.py`: the merged configuration and its hash,
every flow state the run solved (faces, pressure, k, eps, nu_t, the stop record), the
concentration frames at `output_interval` with their times, the budgets, the sensor readings per
step, and a checkpoint group (the simulated time, the step, the five fields, the current flow
state's fingerprint) rewritten every N steps. A restart reopens the file and continues from the
checkpoint; a steady solve checkpoints its faces, pressure and turbulence state every M outer
iterations so a capped run continues from where it stopped rather than from rest.
`FieldHistory.save` stays as ADR-011 decision 8's contract (the Phase 7 reader), its frames
copied into the run file from `frames`.
(2) The probes' way made official: npz plus json per artefact in a run directory. No new
library, but no single record, and the restart reads several files that can drift apart.
(3) HDF5 for the run file and npz for FieldHistory, each written separately: two copies of the
frames.
*Recommendation.* (1). The test that pins it: a run interrupted at a checkpoint and restarted
reproduces the uninterrupted run bitwise, for a steady solve and for a transport march.

**10. VAL-008's case and criterion (section E).**
*Picture.* The plan's row says "observed order within 0.2 of theoretical", for no stated
quantity, scheme or mesh. The transport scheme is QUICK bounded by the UMIST limiter under
forward Euler: second order in space on smooth monotone data, first order at extrema (ADR-011 B),
and first order in time. The probe (T5) advected a Gaussian pulse and a smooth monotone front on
four grids, at a fixed Courant number and at one halved with the spacing.
*Options, ranked.*
(1) The quantity is the relative L2 error of the advected field against the exact translation
(VAL-004's metric); the case is the smooth monotone front on the aligned channel (T5, "front"
and "front_wide"), where the limiter leaves the face value QUICK's wherever the gradient ratio
is near 1 and the theoretical order is the scheme's 2, with the Courant number halved per
refinement so the time error falls as dx^2; the criterion stays "within 0.2 of 2", on the case
T5 measured it on: the wide front (width 0.2 m, 3 widths of travel, its tail at the inlet about
1e-4 of its height), where the pairs whose coarser grids hold 10 and 20 cells across the width
give 1.83 and 1.90, two pairs behind the rule, and the pair at 5 cells gives 1.64. Cells per
width alone does not set the order: the narrow front (width 0.1 m, 10 widths of travel) at the
same 10 to 20 cells gives 1.67, on a grid twice as fine and at half the Courant number, so the
longer travel, over which the limiter's first-order share accumulates, and the inlet's cut of
its tail are what the row must hold fixed. A second row on the Gaussian pulse at the same rule reports its order with
no threshold: a limited scheme's order at an extremum is the limiter's, and the plan's 2 was
never a claim about it. A third row on the stretched mesh (issue 53 D6) reports the front's
order under the uniform-mesh limiter terms.
(2) The Gaussian pulse at a fixed Courant number, as VAL-004 runs: the observed order is forward
Euler's 1 and the criterion "within 0.2 of 2" fails by construction; amend the plan's row to
"within 0.2 of 1".
(3) The plan's row as written, with the theoretical order left to the test: an unstated order is
not a criterion.
*Recommendation.* (1), with the plan's row amended to name the quantity, the case, its
parameters and the Courant rule. The measured orders are T5's; two measured pairs stand behind
the rule, and a fifth grid (dx 0.0025 m, 80 cells per width) would give a third if two are
judged too few.

**11. The cylinder's grid, boundaries and thresholds (section F).**
*Picture.* The paper resolves the body with cells of 0.004 d at its surface and 20 nodes per unit
length, coarsening to 0.5 d in the far field in its default cases and 0.3 d in its Neumann case
(Table IV case 2), with a far-field cell Reynolds number below 30 recommended, which at Re 100
caps the far field at 0.3 d, on 96x94 to 210x202 points, and its Strouhal number
at Re 100 is 0.146 on every grid of its refinement study (Table IV [1]); the onset of shedding is
at Re 51.2 +/- 1 at 5% blockage (Table V [1]). This solver's mesh clusters toward the domain
walls only, so the probe ran a uniform grid of 0.1 d (ten cells across the body, 370x200), the
Neumann outlet 26 units downstream as the paper's Neumann cases, and freestream lateral
boundaries (U = 1, V = 0) in place of the paper's symmetry condition.
*Options, ranked.*
(1) Interior clustering added to `mesh.py` (piecewise geometric spacing between stated
coordinates on each axis, so bands of fine cells cross the body's four faces), the uniform 0.1 d
grid as the ladder's coarse rung, the Neumann outlet at 26 units, freestream lateral boundaries
(at 5% blockage the paper's own blockage change from 5% to 2.5% moved St by under 1.5%, the size
of the lateral condition's effect). Thresholds proposed, OPEN: at Re 100, St within 5% of 0.146
on the clustered grid (0.139 to 0.153; the paper's own spread across outlet treatments and
extents at Re 100 is 0.145 to 0.150, Table II); the bracket, Re 40 steady (the lift signal decays
to zero) and Re 60 shedding (a sustained periodic lift), which contains Re_cr 51.2 +/- 1; the drag
coefficient (1.46 in the paper's case 1) reported, not scored.
(2) A uniform 0.05 d grid (about 296,000 cells, four times the probe's cost per step): 20 cells
across the body with no code change, the far field four times finer than it needs to be.
(3) The convective outlet (the paper's CBC, `du/dt + U du/dx = 0` with a mass correction) as a
new outlet type: the paper's downstream extent falls from 26 to 10 and its iterations per step
roughly halve (Table II); a boundary type to build and test, and the copy rule's known weakness
(air entering a pressure outlet, #61, #69) is what the long march (T3) watches for at 26 units.
(4) A symmetry (slip) lateral condition as the paper's: zero normal velocity with a zero-gradient
tangential value, which the tangential layer already has for pressure outlets; a small addition
to `boundary_registry.py` and `boundary_staggered.py` if the threshold is tight.
*Recommendation.* (1), with (3) and (4) named as the changes to make if the clustered grid misses
the threshold. The thresholds are OPEN for Alex. The identifier proposed is VAL-022, three rows.

**12. The module cut and the build order (section G).**
*Picture.* The plan names `src/scenarios.py`, `src/time_integration.py` and `src/io_manager.py`.
The design above needs, besides them, a warm start and a transient loop in
`solver_staggered.py`, a transient term in `momentum.py`, the scenario schema in `config.py`,
interior clustering in `mesh.py`, and tests and rows for each.
*Options, ranked.*
(1) The plan's three modules with the solver changes in the modules that own the arithmetic:
`scenarios.py` owns the event model and the state sequence; `time_integration.py` orchestrates
both tiers and discretises nothing; `io_manager.py` owns the run file; the transient term is a
predictor option in `momentum.py` and the per-step loop a method of `StaggeredSolver`. Built in
eight ECR-style steps (section G), the housekeeping batch and the warm start first.
(2) A `transient.py` module holding the implicit term and the step loop beside
`solver_staggered.py`: a second solver class to keep in parity with the first, which is what
retiring the collocated solver (PR 29) ended.
*Recommendation.* (1).

## Context
Phase 3 closed with a CONDITIONAL PASS on 2026-10-16: the transport solver advances one class
one step on the face velocities of a converged steady flow (ADR-011 C), the k-epsilon model
gives that flow and its eddy viscosity for the product room on 320x120 (ADR-012; ECR-002 step 8,
`docs/reports/ecr002_step8_product.md`), and the pressure solve is multigrid-preconditioned
conjugate gradients (ADR-013). Nothing yet owns time: the Phase 3 marches were probe scripts
(`docs/reports/probe56/transport56.py` and its predecessors) that loaded a kept flow record,
chose a step from `stable_dt` and looped. The steady solver starts from rest on every call
(`solve_steady` allocates its fields and the inlet turbulence state), keeps no state between
calls and writes nothing; the transport solver keeps a budget and offers `FieldHistory`. Phase
4's deliverables (PROJECT_PLAN, Phase 4) were written before any of the costs above were
measured. This design settles the time integration, the scenario model, the specification and
recovery definitions, the output and restart, VAL-008 and the cylinder's criteria, the modules
and build order, the carried limits and the housekeeping batch, with each decision above.

## A. Time integration (decisions 1 to 5)

### Tier 1: the room
**The state sequence.** A run under tier 1 is a sequence of intervals, each with one steady flow
state. The loader (section B) turns the events into the intervals: the switch times are the
starts and ends of the events that change the velocity layer, and each interval's flow state is
the base boundaries with the active replacements applied. States are keyed by a fingerprint of
the velocity-layer keys of their boundary section, so a door that opens and closes twice has two
states, not four solves. The loader reports the number of distinct states before the run starts,
which is the run's flow cost in units of T1's warm arm.
*Amendment of 2026-10-17 (Alex and the orchestrator, prompt 62's decision 3): the key set is
stated, not inferred.* The velocity-layer keys are `scenarios.FLOW_KEYS`: `type`, `location`, the
four extents, `velocity`, `u_velocity`, `v_velocity`, `obstacle`, `face`, `turbulence_intensity` and
`dissipation_length` (the inlet turbulence changes the flow under k-epsilon), with the segments'
order, since the coverage rule is first match in configuration order. The keys only the transport
reads, `concentration`, `hepa_filtered`, `recirculate_from` and `deposition_surface`
(`scenarios.TRANSPORT_KEYS`), are out; every segment key is in exactly one of the two, which
`tests/test_scenarios.py` checks. *As built (step 2, prompt 62):* contiguous pieces of one segment
(adjacent in configuration order, on one edge or face, end to start, every other flow key equal)
are joined before the fingerprint is taken, so the breach's supply split into three panels keeps
the base's state, as section B's breach has one flow state; with the extents read literally it
would have two. The joined state is the split one to the rounding of the inflow summed per
segment, which the returns' shared velocity carries (1.1e-15 relative on 160x60 and 4.0e-15 on
320x120), not bitwise. The rule was the builder's departure; *ratified (orchestrator and Alex,
2026-10-17, review 62 S8)*. Each key of its equality clause, and pieces on two edges or two
obstacle faces, are pinned by `tests/test_scenarios.py` (prompt 62b, review and test 62 B1).

**The warm start.** `StaggeredSolver.solve_steady` gains `initial: FlowState | None`: the faces,
pressure and, with the turbulence section, k and eps of a state, in place of the fields at rest
and the inlet-uniform turbulence state. The boundary layer's `apply_normal_velocity` then writes
the new state's inlet and outlet faces over the kept ones, as it does over zeros today, and the
loop runs unchanged. The probe did exactly this by patching the two allocations
(`warm59.py`, `solve`); the design makes it an argument. The `error_estimate` rule starts fresh:
its window fills over the first iterations and its conditions are judged on the new state's own
history, so a warm-started solve that is already converged stops when the window says so, not at
once. The first state of a run is solved from rest unless the run file (section D) holds it.

**What a re-solve is, and what it costs.** T1 gives the cost on the product grid: the open-door
state from rest, from the converged closed state, and the closed state back from the open one,
each by outer count and wall time, beside the step 8 closed-door solve from rest. Prediction (a)
is scored on the warm-over-rest ratio in T1, and fails (decision 1's picture): the rest and warm
arms reach the same state (their largest speeds agree to twelve figures, `rest.json` and
`warm.json`), and the warm arm's first residual is about a third of the rest arm's, but the
descent from there to the stop is the same slow geometric one. The flow cost of a tier-1 run is
the number of distinct states times about 0.85 of a steady solve, on this grid with this rule.

**The hand-over.** At a switch the concentration fields are carried across unchanged: the cells
are the same cells, and the budget's current mass is the same number. The faces change, so the
step changes: `time_integration.py` recomputes `stable_dt` for every class on the new faces and
the new eddy viscosity, takes the smallest over the classes (one step for all five, so the frames
and the sensors share a clock), and divides the interval to the next switch into equal steps of
at most that size, as the step 8 march divided its windows. A switch therefore lands on a step
boundary by construction. The eddy viscosity the transport reads (REQ-T13) switches with the
faces. Nothing ramps: the plan's risk "scenario BC modifications create solver instability",
mitigated there by ramping the onset over several steps, was the flow solver's response to a
boundary step change, and under tier 1 that response is the warm-started re-solve, whose
convergence T1 measures; the transport scheme is stable at any face field under its Courant bound
(ADR-011 B's lemma), so a step change in the faces needs no ramp.

**The transport step.** Decision 2. The scenario files state `transport.cfl_number`; the march
runs every class at the one step above. T6 gives the cost per simulated second at the file's
0.1, per class and for all five, measured on the L0 faces and eddy viscosity. The cost scales
with the inverse of the Courant number and with the cell count. T7 gives the time error between
0.1 and 0.4 on the quantity a scenario is scored on, the sensors' recovery times in the
clean-out, and the free-area decay beside the well-mixed line.

**Sources and inlet concentrations in time.** `time_integration.py` asks `scenarios.py` for the
active sources at the step's time, sums them into one rate array per class, and passes it as
`solve_timestep`'s `sources` (ADR-011 decision 5). An event that changes a segment's
concentration keys produces a new `ConcentrationBoundary` for its interval, built from the
interval's merged mapping, which the march reads through `faces_for`; the velocity layer is
untouched, so no re-solve. An event that starts or ends inside a step does not exist: the
loader's intervals are the events' own boundaries.

*As built (step 4, prompt 64, 2026-10-17).* `time_integration.run` builds the hand-over as
stated, with these readings of it. The transport solver reads its concentration boundary when it
is built, so each interval gets a solver of its own on `conditions_for(interval)`, and the
budgets enter it through `restore_budgets`: the same numbers, so the current mass is carried
unchanged, as the step's own first update shows (`tests/test_time_integration.py` steps the first
step after each switch by hand from the step before it and gets the run's step bitwise). Step i
of an interval from t0 is at t0 + i dt and its last step is at the interval's end exactly, so no
time is accumulated and a switch is a step boundary whatever the rounding of n dt. `stable_dt`
reads the faces and the settling velocities only: the eddy viscosity enters the implicit
diffusion, which has no step bound, so "on the new faces and the new eddy viscosity" reads the
faces. The division takes one more step when span / ceil(span / stable_dt) rounds above the
bound, which `solve_timestep` would refuse. The sources are `sources_at` at the interval's start,
constant over it since an event starts and ends on an interval boundary, and a class with no
active source is passed None, the step without one, bitwise. The first state of a run is solved
from rest unless a run file holds it: the restart's own, or, for a new run, another run's file
named by `flows_from` whose configuration agrees on everything the flow solve reads
(`time_integration.flow_inputs`: every section but the scenario, the boundaries, which the
fingerprint compares, the transport, the particles, the sensors and the thresholds, and every
solver key but `dt`, `t_end`, `output_interval`, `checkpoint_interval` and `max_simple_iter`).
Decision 1's "one solve when the closed state is already held in a run file" is this argument.
A state that stops at its cap leaves its steady checkpoint and stops the run
(`FlowNotConvergedError`), so a restart under a larger cap resumes it; it is not written to
`/flow/`, which holds converged states only.

**Decision 2's consequence, stated as a limit.** The flow's own adjustment to a boundary change
takes some turnovers (T6: the turnover is about 5.6 s, the length over the supply speed about
18 s), and the steady extreme replaces that adjustment by the end state at once. For an event
shorter than the adjustment time (a door open for a few seconds) the exchange is overstated,
which is the worst case decision 2 asks for; for one longer (a breach lasting minutes) the
overstatement is confined to its first seconds. Section H carries this beside the Phase 3
limits.

### Tier 2: the cylinder
**The step.** Decisions 3 to 5. Each time step: set the old level(s), then run the outer
iteration (extrapolate the outlets, predict with the transient term, correct, under-relax) until
the velocity step is below the criterion, record the step's imbalance and count, and keep the
faces as the new level. The transient term enters the diagonal and the source of the momentum
equations at every unknown face (the probe's `ImplicitEulerPredictor`), so the pressure
correction's `d = A / a_P` carries it without change and the corrected faces satisfy continuity
to `pressure_rtol` as in the steady solve. The transient loop is a method of `StaggeredSolver`
(`solve_transient`, with the step, the end time, the per-step criterion and cap, and a callback
per step), and `time_integration.py` calls it for a tier-2 configuration as it calls
`solve_steady` for tier 1's states.

**The cost.** T2 gives the outer iterations per step at three criteria, two steps and two pressure
tolerances from one developing field; T3 the cost along a long march from a perturbed start,
where the shedding instability may grow; T4 turns them into hours for one run of the paper's
length. Prediction (b) is scored on T2's counts and T4's hours.

**The step against the period.** The paper's step, 0.025 d / U, resolves a period of 1 / St at
St 0.146 by about 274 steps; 0.05 by about 137 (T4). With BDF2 (decision 3) the larger step is
the proposal, with the smaller as the ladder's check that the Strouhal number does not move.

## B. The scenario and event model (decisions 6 and 7)

**An event.** A name, a start time, a duration, and one of four changes:
- `replace`: a named boundary segment is replaced by one or more segments for the event's
  duration (the door: section B's two segments; a partial filter breach: the supply split into
  a clean part, a breached part and a clean part). Whether the velocity layer changes decides
  the re-solve, by the fingerprint rule of section A; a replacement that keeps every velocity
  key (the breach) is a transport-only event by that rule, with no separate flag to get wrong.
  *Amendment of 2026-10-17 (Alex and the orchestrator, prompt 62's decision 2): a replacement may
  be empty.* `replace` with an empty list removes the named segment for the event's duration,
  and the coverage rule's default, a wall, takes its faces. This is how a tool's exhaust fails or
  a cooling fan stops: section J refuses `wall` on an obstacle face, so without removal neither
  event could be written.
- `set`: keys of a named segment changed in place (the whole-supply breach: `hepa_filtered`
  false and a `concentration` per class).
- `source`: a box (`x_start`, `x_end`, `y_start`, `y_end`) and a `rate` per class, particles
  per cubic metre per second, uniform over the non-SOLID cells whose centres lie inside, or an
  `emission` per class, particles per second per metre of depth, divided by the box's non-SOLID
  area (prompt 47's convention). A point source is a line source in 2D (section H).
- `initial`: the initial concentration per class, uniform over the non-SOLID cells, or the
  thresholds times a factor (the clean-out at 100, the recovery test's challenge [2]). Start 0,
  no duration.

**The door.** Decision 6. The open doorway as two segments on the left edge: `door_in`, a
`velocity_inlet` from the floor to the split height at the exchange velocity, carrying
`turbulence_intensity` and `dissipation_length` (ADR-012 C's inlet rule) and the corridor
`concentration` per class; `door_out`, a `fixed_flow_outlet` from the split height to the door's
top at the exchange velocity plus the offset. The probe's values (0.25 m/s, the split at half the
door, the supply's turbulence keys, offset zero) are stated in `warm59.py` and are assumptions;
the scenario file carries them as keys. The remainder rule of ADR-012 decision 1 keeps the four
returns at their closed-door shares, since the doorway's inflow and outflow cancel; a non-zero
offset reduces the returns' shares by that amount. The seal-leak variant is `door_in` alone over
the seal's extent at a leak velocity, with the returns taking the extra inflow.

**The four reference scenarios as configurations.** Each a file under `configs/` extending
`clean_room_default.yaml` (decision 7), with `t_end`, `transport.cfl_number` and `output_interval`
of its own:
- `scenario_door_leak.yaml`: the door opens at a stated time for a stated duration (section B's
  replacement), corridor air at a configured concentration; two flow states.
- `scenario_filter_breach.yaml`: one panel of the supply (a stated x range) loses its filtration
  for a stated duration: the supply replaced by three segments, the middle one `hepa_filtered`
  false with an upstream `concentration` (the reference value proposed: the thresholds times 100,
  the same factor as the clean-out, since ISO 14644-1 states no Class 8 limit at 0.1 and 0.3 um);
  one flow state.
- `scenario_equipment_dust.yaml`: a source box on a tool top (prompt 47's litho-top source,
  `docs/reports/ecr002_step8_product.md` section 2.5) with a stated emission for a stated
  duration; one flow state.
- `scenario_clean_out.yaml`: `initial` at the thresholds times 100, the supply clean, run until
  within spec (section C) or `t_end`; one flow state.

**The schema, validated at load (REQ-C01, C02).** The `scenario` section: `base` (a file path
relative to the file), `events` (a list), and the run keys above. Every event: `name` a
non-empty string, `start` a finite non-negative number, `duration` a finite positive number
(absent for `initial`), `start + duration <= solver.t_end`, exactly one of the four changes,
bools rejected where numbers are expected, unknown keys refused, as `config.py` does for every
other section. Two events that touch the same segment must not overlap in time; two `initial`
events are refused; a source box must contain at least one non-SOLID cell centre; a
`concentration` or `rate` array has one non-negative finite entry per configured class. Then the
loader builds every interval's merged mapping and validates it through `SimConfig.from_dict`,
so a replacement that overlaps a neighbouring segment or omits a required inlet key fails at
load. The merged base mapping and the validated event list are what `SimConfig` carries
(`config.scenario`), and the file's own sections, when present, override the base's as decision 7
says.

**As built (steps 1 and 2, prompt 62, 2026-10-17).** The schema was built as this section states
it except where these say otherwise, each with its reason.
1. *The reference scenarios extend the room without vents* (Alex, 2026-10-17, prompt 62's
   decision 1), so their results stay tied to Phase 3's validated room and the vent flows stay
   assumptions; vents enter step 1 through a test fixture, `tests/scenario_vent_failures.yaml`
   (a fan stop and an exhaust failure on `configs/clean_room_vents.yaml`). The note of section G
   and the accepted status that the reference scenarios carry vents from the start are
   superseded by this.
2. *The run keys are stated in the file's own sections*, not in the `scenario` section:
   `solver.t_end`, `solver.output_interval` and `transport.cfl_number`, each required in the
   scenario file itself (`config.SCENARIO_RUN_KEYS`), so "of its own" is checked, not assumed, and
   no key has two places to be written.
3. *The event syntax.* `replace: {segment, with: [segments, each with a name]}`, `set: {segment,
   keys: {...}}`, `source: {the box, rate or emission}`, `initial: {concentration or
   threshold_factor}`. A replace or a set names a segment of the merged base; a replacement may
   not take another base segment's name (it may reuse the replaced one's), and two concurrent
   replacements may not give one name twice. Event names are unique (an event is named in the
   run's records), and an initial states `start: 0`.
4. *A source box's non-SOLID cell centre is checked by the manager*, which holds the mesh
   (`ScenarioManager` refuses the box when it is built, before anything runs), as section J's
   "covers at least one face" is checked when the boundary is built.
5. *A fan's two vents stop together.* Stopping a recirculating discharge while its intake still
   draws is not refused by any rule of a file: the intake is then a plain exhaust, and the room
   balances. Prompt 62 expected the pair's balance check to refuse it; that check runs only on a
   discharge that names its intake. The scenario layer refuses an interval in which an intake of
   the base's pair is present and no discharge returns it. Stopping the intake alone is refused
   by the pair's rule (the discharge names a missing segment), and changing the discharge's flow
   alone by the balance check, both through the merged validation. *Ratified (orchestrator and
   Alex, 2026-10-17, review 62 S8)*, with this reason: once the discharge is removed nothing names
   the intake, so the recirculating pair's balance check cannot see a half-stopped fan, and the
   premise of prompt 62 that it would was wrong.
6. *`threshold_factor` needs a threshold for every class*, matched by diameter
   (`SimConfig.class_thresholds`).
7. *`checkpoint_interval`* is a `solver` key, `SimConfig.checkpoint_interval`, not a field of
   section G's `ScenarioSpec`; it arrived with the run file at step 3 (review 64 S7).
8. *Event times are read to one tolerance* (prompt 62b, review 62 S5). Two events written to meet
   (one ends as the other starts) were refused as overlapping when the rounding of
   `start + duration` put the end above the start (0.1 + 0.2 against 0.3), and on two segments
   the same rounding left a sliver interval. `config.EVENT_TIME_RTOL`, 1e-9 of `t_end`, is the one
   instant: an overlap is refused only beyond it, `times()` drops a boundary within it of the one
   before or of `t_end`, `ScenarioSpec.is_active` reads start and end to it (so `active()` and
   the manager's `sources_at` agree with `times()`), and an end past `t_end` by its rounding loads.
9. *An interval's vents are checked on the mesh when the manager is built* (prompt 62b, review 62
   S6): `ScenarioManager` runs `boundary_vents.vent_runs` on every interval's configuration, so a
   replacement vent that covers no face, or a recirculating pair whose flows differ on the faces,
   is refused before a run, as section J refuses a base vent. `scenarios.py` therefore imports
   `boundary_vents` too, beside section G's list.

## C. Specification and recovery (decision 8)

**Within spec.** A class is within spec at a sensor when the sensor's reading (the cell
containing the sensor, as Phase 5's monitor will read it) is below the class's threshold; the
room is within spec when every sensor is, for every class, and stays so to the end of the run
or for a stated hold time. The clean-out's recovery time is the first time the room is within
spec and stays so. The standard's number is reported beside it: the 100:1 recovery time per
sensor and for the free-area mean, defined as the time for the reading to fall to one hundredth
of its initial value, which the clean-out's initial condition makes the threshold itself
(section B). Where the initial level is below 100 times the target the 100:1 time is reported as
not defined, as the standard's alternative by rate would apply [2].

**The well-mixed check.** The free-area mean of each class against C0 exp(-N t), N the supply flow
over the free area per metre of depth (T6), both plotted on a semi-log axis. A well-mixed slice
would lie on the line; the computed mean lies above it where the supply short-circuits to the
returns and contaminated air lingers in the recirculations, and the ratio of the computed 100:1
time to the well-mixed one, ln(100) / N, is the one number the check states. It is a report, not
a pass criterion: no exact solution exists for the slice, and the conservation that would make
it one is already REQ-T05's.

**The field of local recovery time.** Per class, per non-SOLID cell, the last time the cell's
concentration is above the class's threshold, with cells never above it at zero and cells still
above it at the end marked as not recovered. Its maximum over the cells is where the room clears
last; its map is where contamination lingers, which is the stakeholder question of SYSTEM.md
section 1 asked of recovery rather than of deposition. Both outputs are written to the run file
(section D) by `time_integration.py` from the frames and the sensor readings it already holds.

**What Phase 5 reads.** The per-sensor recovery time is the alert's clearing time, the companion
of REQ-A04's detection latency; the local recovery field gives the clearing time a sensor would
have reported at any position, which is REQ-A05's placement question for the clean-out as the
detection latency is for the other three scenarios.

*Amendment (step 4, prompt 64, 2026-10-17; the orchestrator's decisions 1 and 2 of that date
and the departures the build made).*
1. **The local recovery field is updated every step, not from the frames.** Read from the frames
   its time resolution would be `output_interval` steps, which the output cadence should not set.
   Per class and per non-SOLID cell, the last step time the cell was at or above its threshold is
   kept as an array during the march and carried in the march's checkpoint, with the other running
   readings: each sensor's and the free-area mean's first time at the 100:1 level, and the start of
   the current within-spec stretch per class and for the room (`time_integration.RecoveryTracker`).
   A restart therefore continues them bitwise.
2. **The hold time.** `scenario.hold_time`, seconds, optional. With it the run stops once the room
   has been within spec for that long and the recovery time is the start of the hold; without it
   the run goes to `t_end` and the recovery time is the start of the last within-spec stretch that
   reaches `t_end`. The loader refuses a hold without an initial event (a room that starts clean is
   within spec from t = 0 and the hold would end the run before any event) and one not below
   `t_end`. `scenario_clean_out.yaml` holds one air change of the slice, 1/N = 17.52 / 3.15 =
   5.562 s, the arithmetic in its comment.

The departures, each with its reason:
- *A sensor reads the cell containing it, the one below it or to its left when it sits on a face*
  (to `SENSOR_FACE_RTOL`, 1e-9 of the room's extent). The room's sensors sit on faces on every grid
  that puts its positions on faces, so "the cell containing the sensor" needs a rule there. T7's
  probe took the nearest centre, which on a face is decided by rounding: on 80x30 it puts
  above_gap_1 to the right of its face and the other three sensors to the left. The rule reads the
  cells T7 read on 160x60, and on 320x120.
- *"Above" is "not below".* Within spec is a reading below the threshold, so the local field
  counts a cell exactly at its threshold as not recovered.
- *A class's recovery time* is the room's rule applied to that class's sensors; the room's is the
  latest of the classes', since the room's within-spec stretch that reaches the end starts when the
  last class's does.
- *N is the inflow through the domain's edges of the first interval's flow state, over the free
  area* (`time_integration.air_change_rate`). A vent's discharge is the room's own air and is not
  counted.
- *The outputs a run cannot reach are NaN*, and the summary gives the ratio's lower bound, the run's
  end over ln(100) / N, when the free-area mean was still above its level at the end.

What the coarse runs found (80x30, `results/builder64/coarse/`, the summaries
`clean_out.txt` and `clean_out_nohold.txt`): every sensor of every class is below its threshold
within about one well-mixed 100:1 time, so the hold of one air change stops the run a few seconds
later; at that stop the free-area mean has not fallen to its 100:1 level and cells at or above
their thresholds remain in every class, so section C's ratio is not defined by the held run and the
local field's maximum is not recovered. Run without the hold to `t_end`, the mean reaches its level
at about three times the well-mixed time and cells above threshold still remain at 120 s. The four
sensors clear long before the room's air does. Whether the hold of decision 2 should end a
clean-out on the sensors alone, or the run should continue until the ratio and the field are
defined, is a question for the orchestrator; the build follows decision 2 as written.

*Amendment (prompt 66, 2026-10-17; review 64 S1, the orchestrator's decision 2 of prompt 64
withdrawn).* **The reference clean-out runs to its end.** `scenario_clean_out.yaml` has no
`hold_time` and runs to `t_end` 240 s. The sensors clearing is not the room clearing: on 80x30
the hold of one air change stopped the run at 31 s, when the free-area mean had not reached its
100:1 level and 18% of the free cells were still above threshold, so the ratio and the local field,
two of the clean-out's three outputs, were not defined by the reference run. Without the hold the
coarse mean reached its level at about 78 s, three times the well-mixed 25.6 s, and 2% of the
cells were still above threshold at 120 s. 240 s is about 1.5 hours on the product grid at T6's
cost scaled to Courant 0.4. `scenario.hold_time` stays as a feature; the suite pins it on the held
run prompt 64 made (to 120 s, held one air change).

## D. Output, checkpointing and restart (decision 9)

**The run file.** One HDF5 file per run, written by `io_manager.py` (h5py, already required):
- `/config`: the merged mapping as YAML text, its SHA-256, the commit, the start time, the
  machine and the number of concurrent processes, as the probe records carry them.
- `/flow/<fingerprint>/`: for every flow state solved, `u_faces`, `v_faces`, `p`, and with the
  turbulence section `k`, `eps`, `nu_t`, with the stop reason, the outer count, the wall time,
  the stage seconds and the rule's readings at the stop as attributes.
- `/transport/`: `steps`, `times` and `C_<k>` frames at `output_interval` (the `FieldHistory`
  keys, so Phase 7's reader has one format), the budgets per class per frame, the sensor
  readings per step per class, and the two recovery outputs of section C when the scenario
  is a clean-out.
- `/checkpoint/`: the simulated time, the step index, the five fields, the fingerprint of the
  current flow state and the budgets, rewritten every `checkpoint_interval` steps (a new
  `solver` key, validated like `output_interval`), so a crash or a cap loses at most that many
  steps.
- `/steady_checkpoint/`: during a steady solve, every `checkpoint_interval` outer iterations,
  the faces, pressure and turbulence state, with the outer count: a capped solve (prompts 49
  and 52) is continued with `initial=` from here, which is the warm start of section A applied
  to the same state. *Note (step 0, prompt 60, 2026-10-16):* the warm start cannot continue a
  solve bitwise, since it rebuilds nu_t from k and eps (the under-relaxed nu_t the next outer
  iteration reads is not that) and starts the rule's window empty (so it cannot stop before
  100 more outer iterations, wherever the interrupted solve would have stopped). Step 0
  therefore builds the continuation as its own keyword, `solve_steady(resume=checkpoint)`, with
  a `SteadyCheckpoint` that carries the relaxed nu_t, the rule's window, the residual history,
  the cap hits and the outer count, and keeps `initial=` as section A's warm start. The group
  holds the checkpoint whole (`docs/reports/adr014_step0.md`, section 7).
  *Note for step 3 (review 60, prompt 60b, 2026-10-16):* two things step 3 builds on top of
  step 0's `write_steady_checkpoint` and `solve_steady(resume=)`. (1) The rewrite must be
  crash-safe: step 0 deletes the old group and then writes the new one, so a process killed in
  between leaves no checkpoint or a partial one, which is the loss the periodic rewrite is
  there to prevent. Step 3 writes the new group under a temporary name and swaps it in, and
  never deletes first. (2) A resume should refuse a changed `eddy_viscosity=` argument, as it
  refuses a changed grid: the laminar solve's prescribed field is an input of the solve that no
  checkpoint carries and no configuration hash covers, so a resume under another field is not a
  continuation.

**Restart.** `time_integration.run(config, restart=path)`: the configuration hash must match the
file's, the flow states in the file are reused, the march resumes from the checkpoint's step
with the checkpoint's fields, and the frames after it are appended. The test that pins it is
section D's: interrupted at a checkpoint and restarted, a run reproduces the uninterrupted run's
fields bitwise, for a steady solve and for a transport march, which the deterministic arithmetic
of both solvers allows.

*Amendment (step 3, prompt 63, 2026-10-17; the orchestrator's three decisions of that date
and the departures the build made).*
1. **The checkpoints live in a checkpoint file beside the run file, replaced atomically.** The
   note for step 3 asked for a crash-safe rewrite by writing a new group under a temporary name
   and swapping it in. Inside one HDF5 file that protects little: HDF5 keeps no journal, and a
   process killed while the file is open for writing can leave the whole file unreadable, the
   frames included. So `/checkpoint/` and `/steady_checkpoint/` live in `<run>.checkpoint.h5`
   (the run file's path less its suffix), and each checkpoint writes that file whole under
   `<run>.checkpoint.h5.tmp` (mode "w", so a stale one an earlier crash left is truncated,
   never read), copies the group it does not replace across, closes both files and moves the
   new one over the old with `os.replace`, which is atomic on one volume, Windows included.
   The run file is opened, appended and closed at each output, so a crash between writes
   leaves it readable with every frame written before. "One HDF5 file per run" above becomes
   one run file plus its checkpoint file. Step 0's two module functions, which wrote into the
   run file in place, are replaced by `RunFile`'s methods; step 0's files are not read.
2. **The restart's configuration hash leaves out the keys a continuation exists to change.**
   Prompt 60's continuation test raises the cap, and a capped march is continued by raising
   `t_end`, so a hash over the whole merged mapping would refuse the restarts this section
   exists for. `/config` keeps the SHA-256 of the whole text as the records carry it, and
   beside it the restart hash, the same over the mapping less
   `io_manager.RESTART_EXCLUDED_KEYS`: `solver.max_simple_iter`, `solver.t_end`,
   `solver.checkpoint_interval`. Every other change is refused, the message naming the first
   key, in the file's order, that changed, was added or was removed. Key order counts, since a
   boundary segment's place in the order is part of its meaning.
3. **A resume refuses a changed `eddy_viscosity=`** (the note's second point).
   `SteadyCheckpoint.eddy_viscosity_sha256` is the SHA-256 of the prescribed field's bytes, or
   None, and `solve_steady(resume=)` refuses a mismatch either way, None against a digest
   included.

The departures, each needed by the restart as step 4 will call it: the cadence is handed out by
the solver, `solve_steady(on_checkpoint=...)`, called every `checkpoint_interval` outer
iterations with the checkpoint a solve capped there leaves (counted from a resumed solve's
outer, never on the converging iteration); the steady checkpoint carries the fingerprint of
the flow state it solves for, so a restart can tell which solve it continues; a restart drops
the frames and sensor readings written after its checkpoint (`RunFile.discard_after`) before
appending again; the restored budgets enter the transport solver through
`TransportSolver.restore_budgets`, since its `budget` is written by `solve_timestep` only; the
sensor readings are appended in batches at each output rather than per step. Section C's two
recovery outputs are step 4's to compute and write. The march's checkpoint holds the class
fields, the budgets, the step and the time and nothing of the vents: a recirculating discharge
reads its intake from the field the step starts from, which `tests/test_io_manager.py`'s march
restart shows on a vented room with a recirculating pair.

*Amendment (step 4, prompt 64, 2026-10-17; the orchestrator's decision 3 of that date, review 63
S1 to S3 and the builder's note, and the departures the build made).*
1. **The sensor readings held for the next frame are appended before each checkpoint is written**,
   and a step's frame is written before its checkpoint, so a restart leaves no gap and re-emits no
   frame whatever the two intervals are (review 63 S1). The readings are written from step 0, the
   initial field's, and the frames at step 0, every `output_interval` steps and the last step; the
   march's checkpoint every `checkpoint_interval` steps and at the last.
2. **Section C's outputs are a top-level `/recovery/` group**, not part of `/transport/` as the
   list above put them, so `discard_after` does not touch them (review 63 S3); written once at the
   end of the run (`io_manager.RecoveryRecord`) and replaced by a restart that runs to a new end.
3. **The restart check recomputes the file's hash from its stored `/config` text** under the
   excluded-key set in force now (review 63 S2), so a key added to `RESTART_EXCLUDED_KEYS` widens
   what an existing file's restart accepts, as SYSTEM.md says; `restart_sha256` stays as the record.
4. **`SimConfig.merged_mapping()`** is the mapping a configuration loaded, which `write_config`
   stores: a file as read, a scenario as its merged base followed by its scenario section as the
   file states it, so the events and the hold are in the restart hash.

The departures, each needed by the run as built:
- `run(config, run_file, *, restart, flows_from, on_step, provenance)`: section G's contract had
  no place for a new run's path, and none for the provenance `write_config` requires; a new run
  refuses an existing run file or checkpoint file.
- A restart that finds no march checkpoint (none written yet, or no `checkpoint_interval`) drops
  `/transport/` (`RunFile.discard_march`) and marches again from step 0 on the run file's states.
- A restart under a later `t_end` whose checkpoint lies inside the last interval of the earlier
  plan divides the rest of that interval again from the checkpoint's time; a restart under the same
  configuration finds its checkpoint on the plan and is the uninterrupted run bitwise.
- The march's checkpoint carries the recovery tracker's arrays as named float64 arrays
  (`RunFile.checkpoint(..., recovery=...)`), so io_manager knows nothing of the tracker.
- The stopping rule's own readings are not exposed by the solver; the readings recorded with a flow
  state are those it shows: the last residual, the worst and the signed imbalance over the
  non-SOLID cells, and the pressure cap hits.

**Tier 2.** The transient run's file holds the faces and pressure every `output_interval` steps
with their times, the per-step counts and imbalances, and the lift and drag on the obstacle
(section F), and checkpoints the two old levels for BDF2.

## E. VAL-008, grid convergence (decision 10)

**What the row measures.** The observed order of the transport scheme's advection under
refinement: the relative L2 error against the exact translation on a sequence of grids, the
order between successive grids as log2 of the error ratio, on the aligned channel of T5 with
the time step's Courant number halved per refinement so the forward Euler error falls as dx^2.
Three rows: the smooth monotone front (the criterion row, theoretical order 2, "within 0.2", as
T5's wide front: width 0.2 m, 3 widths of travel on the 2.0 by 0.4 m channel at the supply
speed, the front 4.5 widths from the inlet at the start, grids of 0.02 and 0.01 m or finer, the
order read between pairs whose coarser grid holds at least 10 cells across the width); the
Gaussian pulse (reported, no threshold: the limiter is first order at an extremum, ADR-011 B,
and its observed order on a pulse is between the two); the front on the stretched mesh
(reported, issue 53 D6: the limiter's three uniform-mesh terms on a stretched mesh). T5 holds
the probe's measurements for every row and both schemes, with upwind as the control at order 1.
What T5 shows about the rule: on the wide front the pairs with 10 and 20 cells across the width
on their coarser grid pass (1.83, 1.90) and the pair with 5 does not (1.64); the narrow front at
the same cell counts does not (1.67 at 10 to 20), though its grid is twice as fine and its
Courant number half, because it travels 10 widths to the wide front's 3, over which the
limiter's first-order share at the shoulders accumulates. The row therefore fixes the travel in
widths with the resolution, and reads its order between the two finest grids. The front's upstream tail must be small against the error at the
inlet (both probe fronts start 4.5 widths from it, a tail of about 1.2e-4 of the height), or
the inlet's zero cuts the profile and the error floors, which the first placing of the wide
front showed.

**Why not the pulse at a fixed Courant number.** Under forward Euler the time error is O(dt) and
dt = c dx / U, so at a fixed c the error falls as dx and the observed order tends to 1 whatever
the space scheme (T5, the "fixed" rows); ADR-011 C recorded this when it named Heun's method.
The plan's "theoretical" must be the space scheme's, and the time step must be refined faster
than the mesh to see it.

**The amendment proposed.** PROJECT_PLAN's VAL-008 row: "Observed order of the limited transport
scheme on a smooth monotone front (width 0.2 m, 3 widths of travel, 4.5 widths from the inlet at
the start), Courant number halved per refinement, within 0.2 of 2 between grid pairs whose
coarser grid has at least 10 cells across the front's width; the pulse and the stretched-mesh
rows reported" (section G's step 5). REQ-N02's text stands; its
rationale's "second-order scheme should reduce error by 4x" applies to the front row.

**As built (prompt 65, 2026-10-17, branch phase4/step5-val008).** `tests/test_grid_convergence.py`
carries the three rows and the control. The records below are the lines the file prints under
`pytest tests/test_grid_convergence.py -s`, except the two four-grid rows, which are the same
functions called on levels 0 to 3 (`halved_run` and `error_of`, grids 0.04 to 0.005 m). The file
takes about a minute in all (55 s and 69 s in two runs, nothing else running).

| Row | Scheme | Grids (dx, m) | Courant numbers | Relative L2 per grid | Observed orders | T5's orders | Asserted |
|---|---|---|---|---|---|---|---|
| front, the criterion row | umist | 0.02, 0.01 | 0.05, 0.025 | 3.0971e-03, 8.6993e-04 | 1.83 | 1.83 | within 0.2 of 2 |
| front, the control | upwind | 0.02, 0.01 | 0.05, 0.025 | 4.3922e-02, 2.4268e-02 | 0.86 | 0.86 | within 0.2 of 1 |
| front, four grids (not in the suite) | umist | 0.04, 0.02, 0.01, 0.005 | 0.1, 0.05, 0.025, 0.0125 | 9.6745e-03, 3.0971e-03, 8.6993e-04, 2.3316e-04 | 1.64, 1.83, 1.90 | 1.64, 1.83, 1.90 | none |
| front, four grids (not in the suite) | upwind | 0.04, 0.02, 0.01, 0.005 | 0.1, 0.05, 0.025, 0.0125 | 7.4355e-02, 4.3922e-02, 2.4268e-02, 1.2838e-02 | 0.76, 0.86, 0.92 | 0.76, 0.86, 0.92 | none |
| Gaussian pulse, reported | umist | 0.04, 0.02, 0.01 | 0.1, 0.05, 0.025 | 1.3492e-01, 6.0082e-02, 2.7121e-02 | 1.17, 1.15 | 1.17, 1.15 | none |
| front on the stretched mesh, reported | umist | 0.04, 0.02, 0.01 | 0.1, 0.05, 0.025 | 1.0417e-02, 3.5004e-03, 1.0747e-03 (unweighted cell norm) | 1.57, 1.70 | not in T5 | none |

The relative L2 is `validation.metrics.relative_l2`, a norm over the cells with no area weight.
On the uniform rows that is the area-weighted norm scaled; on the stretched row, whose end cells are
0.50 dx wide against about 1.75 dx at the centre, it is not (test 65 S1). Nothing is asserted on
that row.

Every error and order T5 states for the grids run reproduces to the digits it states.
The four-grid rows are reproduced by calling the test module from the repository root:
`from tests import test_grid_convergence as t`, then `t.halved_run(t.FRONT_CASE, k, scheme)` for
k in 0 to 3 and `t.orders_of` over `t.error_of` of each run (about 95 s for UMIST).

Decisions taken in the build. Each is the builder's and open to reversal.

1. *The criterion row's grids are 0.02 and 0.01 m.* Section E names these as the least and says
   "or finer". They give one pair, whose coarser grid holds the 10 cells across the width the rule
   asks, and the pair at the rule's limit is the one nearest the bound. The row takes about 10 s and the
   control under 5 s, so the row stays in the suite, with no slow marker and no harness entry. The pair
   from 0.01 to 0.005 m, the finest in T5, is in the four-grid rows above and is not asserted: the
   0.005 m grid alone takes 82 s, and the three-grid UMIST row 93 s. Gating it needs a slow
   mechanism the repository lacks. `pyproject.toml` registers no slow marker, and the harness
   (`scripts/benchmark.py`) records steady solves (outer iterations, solver parameters, a trajectory
   in each row), not marches, so a transport row needs a record family of its own there.
2. *The control's band is 0.2 about 1,* the criterion's own width. Section E says near order 1 and
   gives no number.
3. *The stretched row.* T5 has no front on a stretched mesh: its stretched rows are the oblique pulse
   with y clustered at a fixed ratio of 1.05. A front uniform in y is blind to y clustering, so the
   row clusters x, the axis it travels along, with `min_wall_spacing` at half the grid's uniform
   spacing: cell widths run from dx / 2 at the inlet and the outlet to about 1.75 dx at the centre,
   where the front travels. The ratio of adjacent widths then falls under refinement, as on a mesh
   mapped smoothly from a uniform one. A fixed per-cell ratio would not: the centre cells would stop
   shrinking and the end cells would collapse (ratio 1.05 on 200 cells puts the end cell at
   0.38 mm). The row asserts that the mesh is stretched, so a mesh that came back uniform cannot
   stand in for it.
4. *Planted defects.* Each was applied to a copy of the tree and the file run against it (the
   pulse row left out, for time). The first three are the build's checks and are kept as tests
   in the file, with the order each reads there.

| Defect planted | Site | Tests that fail | Order read |
|---|---|---|---|
| limiter returns the upstream cell | `limited_face_values` | the criterion row; the control holds | 0.86 |
| Courant number held at its coarse value | `courant_at` | the criterion row, the Courant unit test | 0.99 |
| exact front one cell off | `error_of` | the criterion row | 1.01 |
| solver reads every scheme as upwind | `TransportSolver.__init__` | the criterion row; the control holds | |
| limiter's psi capped at 1 in place of 2 | `limited_face_values` | the criterion row | |
| order by natural log | `observed_order` | the criterion row, the control, the order unit tests, the limiter plant | |
| front 1.5 widths from the inlet | `FRONT_START` | the criterion row, the tail unit test | |
| criterion on the 5-cell pair | `CRITERION_LEVELS` | the criterion row, the control, the 10-cell unit test, the limiter plant | |
| control runs UMIST | the control test | the control | |
| each kept plant's defect removed | the three plant tests | that plant test | |
| exact averages by point values | `front_cell_averages` | the quadrature unit test only | |
| stretched row on a uniform mesh | `advance` | the stretched row's guard | |

The point-value plant leaves the row's order where it was, since the profile enters the initial and
the exact field alike; the quadrature test is what pins the closed form.

## F. The square cylinder (decision 11)

**The case.** A unit square cylinder in a channel of height 20 (5% solid blockage), the inlet 10
units upstream of the body's front face at U = 1, V = 0, the body's faces no-slip, Re = U d / nu
= 100 with rho = 1 and mu = 0.01 [1, sections 2 and 2.1]. The outlet: the paper's Neumann
condition is this solver's pressure outlet (the zero-gradient copy, ADR-012 decision 1's note),
which the paper found needs about 26 units downstream, against 10 with its convective condition
[1, Table II and section 4.1]. The lateral boundaries: the paper's symmetry condition
(dU/dy = 0, V = 0); here a `velocity_inlet` with `u_velocity` U and `v_velocity` 0, a freestream
condition (option (4) of decision 11 builds the paper's). The configuration is
`cylinder59.py`'s `cylinder_raw`, to become `configs/validation_cylinder.yaml`.

**The runs (VAL-022, proposed).** Three rows at the paper's step or decision 3's: Re 40, below
onset, run until the lift's oscillation decays (steady); Re 60, above onset, run to a sustained
periodic lift; Re 100, run to Sohankar's length (6,300 steps of 0.025, T4) with the Strouhal
number from the lift signal's spectrum over the last ten periods. The onset bracket is the
Re 40 and 60 rows together: the paper's Re_cr is 51.2 +/- 1 at this blockage (Table V [1]) and
its own runs at Re 50 became steady and at 55 shed (section 4.4 [1]). Lift and drag are the
pressure and viscous forces on the obstacle's faces, which the staggered layout gives directly
(the pressure at the face cells, the shear from the wall stencil), a small instrument in the
tier-2 step.

**Thresholds, OPEN.** St at Re 100 within 5% of 0.146 on the clustered grid; the bracket as
stated; Re 60's St and the drag coefficients reported. The paper's Re 100 values across its
cases are 0.145 to 0.150 (St), 1.45 to 1.49 (C_D) (Tables II and IV [1]). Whether 5% is the
number is Alex's; the probe's uniform grid is not expected to meet it, which is why the clustered
grid is the row's grid and the uniform one the ladder's coarse rung.

**The grid.** Decision 11: interior clustering in `mesh.py`, a list of breakpoints per axis with
the spacing at each and geometric growth between, validated like the existing stretch keys; the
cylinder's file states fine spacing at its four faces and coarsening to the far field, as the
paper's hyperbolic tangent did [1, section 2]. The far-field cell is at most 0.3 d, the paper's
recommended cell Reynolds number of 30 at Re 100 [1, section 4.3; its Neumann case runs 0.3 d,
Table IV case 2].

## G. Modules, interfaces and the build order (decision 12)

**Contracts (draft).** They replace SYSTEM.md section 4's planned `ScenarioManager`:
`get_active_sources(t)` becomes `sources_at(t)`, one rate array per class, and
`get_bc_modifications(t)` becomes `intervals()` and `flow_states()`, since a boundary
modification is an interval's flow state, not a per-call mapping; section 3.2's row that says
those interfaces are unchanged goes with it. SYSTEM.md carries the contracts below, marked
planned, when this ADR is accepted, and as built at the Phase 4 gate.

```
config.py
    SimConfig.scenario: ScenarioSpec | None    # the merged base, the events, the run keys
    ScenarioSpec: base, events: tuple[EventSpec, ...], hold_time
    SimConfig.checkpoint_interval: int | None  # a solver key (review 64 S7)
    EventSpec: name, start, duration, change: Replace | Set | Source | Initial

scenarios.py
    ScenarioManager(config, mesh)
        intervals() -> list[Interval]          # (t_start, t_end, flow_fingerprint, merged mapping)
        flow_states() -> dict[str, dict]       # fingerprint -> the velocity-layer mapping
        sources_at(t) -> dict[int, ndarray]    # class -> [ny, nx] rate, summed over active sources
        conditions_for(interval) -> ConcentrationBoundary
        initial_fields() -> dict[int, ndarray]

solver_staggered.py
    FlowState(u, v, p, k, eps, nu_t, stop)     # frozen, read-only arrays
    solve_steady(on_iteration=None, *, eddy_viscosity=None, initial: FlowState | None = None)
    solve_transient(dt, t_end, *, initial, step_tol, step_cap, on_step=None) -> FlowState
    flow_state -> FlowState                    # the last solve's, converged or not

momentum.py
    predict(u, v, p, mu_eff=None, wall_mu=None, transient=None)
        transient: (dt, u_old, v_old) or for BDF2 (dt, (u_n, u_nm1), (v_n, v_nm1))

time_integration.py
    run(config, *, restart=None, on_step=None) -> RunRecord
        tier 1: the intervals, the flow states (cached, warm-started), the march
        tier 2: solve_transient with the recorders of section D

io_manager.py
    RunFile(path).write_config / write_flow_state / write_frames / checkpoint / read_checkpoint
```

**Dependencies and the system map.** `scenarios.py -> config, mesh, boundary_concentration,
boundary_registry, particles`; `time_integration.py -> config, mesh, boundary_staggered,
boundary_concentration, solver_staggered, solver_transport, scenarios, io_manager, particles`
(and `monitor` in Phase 5); `io_manager.py -> solver_transport (FieldHistory), staggered
(FaceVelocities), turbulence (TurbulenceState)`. No cycle: `io_manager` imports solver types and
nothing imports `time_integration`. The generated matrix of SYSTEM.md section 3.1 gains three
rows; the cascade rows of 3.2 for `scenarios.py`, `time_integration.py` and `monitor.py` exist
and are rewritten when built; `solver_staggered.py`'s row gains `initial` and
`solve_transient`, `momentum.py`'s the `transient` argument (bitwise the steady path when None,
as `mu_eff` is), `mesh.py`'s the breakpoints.

**The build order, ECR-style steps, each with its tests and its row.**
0. The housekeeping batch (section I) and the warm start: `solve_steady(initial=)`, `FlowState`;
   tests: a warm start from a converged state stops within the rule's window with the faces
   unchanged to rounding, a warm start from a wrong shape is refused; the steady checkpoint and
   its bitwise continuation (section D). *Note (step 0, 2026-10-16):* the faces move by the
   iteration error the first solve's stop left, which the rule bounds by
   `iteration_error_tol` times the velocity scale, so the test holds them to that bound, not to
   rounding (`docs/reports/adr014_step0.md`, section 7); the continuation is `resume=`, not
   `initial=` (section D's note).
0b. The equipment vents (section J, the amendment of 2026-10-16): a boundary segment on an
   obstacle face, as a `velocity_inlet` (a cooling discharge) or a `fixed_flow_outlet` (a
   process exhaust), read by the velocity layer, the momentum and pressure steps, the wall
   treatment and the concentration boundary; the configuration keys and their rejection paths
   in `tests/test_config.py`; the tests section J lists. Before step 1, so the scenario schema
   and the reference scenarios are written with vents from the start. *Built 2026-10-16 (prompt 61),*
   *with the departures section J lists at its end and the recirculated discharge of its amendment.*
1. The scenario schema and inheritance in `config.py`: `tests/test_config.py` gains the rejection
   paths listed in section B, one test per rule, and the four reference files load. *Built
   2026-10-17 (prompt 62), with the departures section B lists after its schema.*
2. `scenarios.py`: intervals, fingerprints, sources, conditions; `tests/test_scenarios.py`:
   event activation at its start and expiration at its end, overlap refused, two openings give
   two states not four, a source's rate sums to its emission over the box. *Note (step 0b,
   2026-10-16, review 61 S1 and S2):* in a room with a recirculating pair the removal is
   `MassBudget.outflow - MassBudget.inflow`, since the budget books the intake as outflow and
   the discharge as inflow; the HEPA capture of a filtered discharge is inside that difference
   and `deposited` is not. The gross terms are what close `MassBudget.residual()`, so a step
   that reads removal reads the difference, as `docs/reports/probe61/transport61.py` does.
   `vent_runs` and `ROOM_EDGE_OF_FACE` move to a small module both boundary layers import,
   with `ROOM_EDGE_OF_FACE` beside the registry, when this step next edits the dependency map.
   *Built 2026-10-17 (prompt 62):* `src/scenarios.py` with section A's key set and section B's
   notes; `src/boundary_vents.py` holds `VentRun`, `VentFaces`, `vent_runs` and
   `vent_face_masks`, and `ROOM_EDGE_OF_FACE` is `boundary_registry`'s, which no longer imports
   `staggered`. `staggered -> config` stays: `obstacle_face_inputs` takes an `ObstacleSpec`.
   Every hash of record unchanged.
3. `io_manager.py`: the run file, the checkpoint, the restart; `tests/test_io_manager.py`: the
   bitwise restart for a march and for a steady solve, the configuration-hash mismatch refused.
   *Built 2026-10-17 (prompt 63), with section D's amendment of that date: the checkpoints in a*
   *checkpoint file replaced atomically, the restart hash less the continuation keys, the*
   *eddy viscosity digest. Every hash of record unchanged.*
4. `time_integration.py`, tier 1: `tests/test_time_integration.py`: the hand-over (the
   concentration field and the budget's current mass carried across a switch unchanged), the
   step divides every interval, the sources enter the budget's source slot, the three
   transport-only scenarios run on a kept flow state end to end on a coarse grid; the clean-out
   with section C's outputs.
   *Built 2026-10-17 (prompt 64), with the amendments of sections A, C and D of that date:*
   *`src/time_integration.py`, `scripts/run_scenario.py`, `scenario.hold_time`,*
   *`SimConfig.merged_mapping()`, the `/recovery/` group. The restart after a switch, the*
   *recovery field included, is the uninterrupted run bitwise on its first run. Every hash of*
   *record unchanged.*
5. VAL-008 (`tests/test_grid_convergence.py`, section E) and the plan's row amended.
6. Tier 2: the transient term in `momentum.py` (bitwise the steady path when None), BDF2 and
   the step loop in `solver_staggered.py`, interior clustering in `mesh.py`, the lift and drag
   instrument, `configs/validation_cylinder.yaml`; VAL-022's three rows in
   `tests/test_cylinder.py`, the Re 100 row marked slow and run by the harness as VAL-017 is.
7. The four reference scenarios run on the product grid with their records in the run files,
   the recovery outputs, and the Phase 4 gate report with this document's planned-against-built
   table.

## H. The carried limits
Phase 3's gate conditions, each with the scenarios it bears on, so Phase 4's outputs carry them:
- *Two-dimensional slice.* All four scenarios. The door's exchange is per metre of depth, the
  dust release a line source, the clean-out's air change rate the slice's (T6: about 650 per
  hour), not the room's. Sensor latencies and recovery times are comparative between layouts
  and sensor positions, not absolute.
- *Turbulence over the equipment tops about 30% high (VAL-020).* The dust release, whose source
  stands on a tool top; the clean-out's recovery over the tops; the door scenario where the
  doorway jet meets the near-door equipment. Mixing and deposition there read high by a similar
  factor in every scenario, so comparisons stand.
- *Separation bubbles short on practical grids (VAL-019).* The door scenario (the doorway jet's
  separation at the opening's edges and the recirculation beside the door wall, where the L2
  cycle of step 8 also sat) and the clean-out's lingering zones behind the equipment: both are
  recirculations the model shortens.
- *Turbulent spreading unvalidated against measurement (VAL-021).* Every scenario's plume; the
  Schmidt-number range of VAL-018 is the stated spread, and a scenario's sensor latency is
  reported at Sc_t 0.7 with the range noted.
- *L2's cycle on 160x60.* The scenarios run on 320x120; a Phase 5 layout comparison on the
  coarse grid carries it.
- *Of this design.* The steady extreme (section A) overstates short events; the door's exchange
  velocity and split are assumptions until a measurement sets them (section B); the cylinder's
  lateral boundary is the freestream condition unless option (4) is built (section F).
- *The equipment vents carry air only (section J, 2026-10-16).* A cooling discharge leaves a
  tool face at the temperature of the room in this model: the heat it carries, and the rising
  plume that heat makes over a tool, wait for a temperature equation and a buoyancy term in a
  later step. Where a discharge is warm in the room it models, the computed jet is the
  isothermal one and the mixing above the tool is the forced jet's alone. The discharge carries
  its configured concentration, not the mean of the air its intake drew in: the intake and the
  discharge are two segments the transport does not link, as the supply is not linked to the
  returns (ADR-011 decision 3, no recirculation). *Amended 2026-10-16 (prompt 61, section J's
  amendment):* a discharge that names its intake in `recirculate_from` carries the intake's
  concentration, lagged one transport step; the limit stands for a discharge that does not.

## I. The efficiency batch (step 0's housekeeping)
Listed as the first build step's housekeeping, each with its location:
- `TurbulenceBoundary.largest_inlet_eddy_viscosity` (`src/turbulence.py`): no caller in `src/`
  since condition (e)'s scale moved to the FLUID median on 2026-10-11 (SYSTEM.md, Document
  History); remove with its tests (`tests/test_wall_functions.py` and
  `tests/test_coupled_solve.py`) and its contract line in SYSTEM.md section 4.
- The constants named by tests 55 and 56: `validation/impinging_jet.py`'s `FREE_JET_H`,
  `NARROW_FREE_JET_H`, `IMPINGEMENT_H` and `PLATE_MEDIAN_HALF_WIDTH_W` pinned by no test (test 55
  S1); `tests/test_benchmark.py`'s product-value test, whose planted field cannot tell a median
  from a mean or the hood-bench top from 2.0 m, and the missing `load_preset("val018_320x120")`
  assertion (test 56 S1).
- Docstring and comment widths: `ruff` ignores E501 (`pyproject.toml`, "line length handled by
  formatter") and the formatter does not wrap docstrings, so lines over the configured 88 in
  docstrings and comments are reflowed by hand in the modules the batch touches.
- The reports of ECR-002 steps 5, 6 and 48 (`docs/reports/ecr002_step5_convergence.md`,
  `ecr002_step6_product_coupled.md`, `ecr002_step6_comparative.md`, `ecr002_step6_variants.md`):
  prompt 59 names `|` characters inside table cells; a scan at 809d1c6 for a backslash-escaped
  pipe found none, so the step either locates the cells by rendering the tables or strikes the
  item with that finding.
- `docs/reports/probe43/field43.py` runs `git rev-parse` at import to find the main checkout's
  `results/`; move the call into a function, as `warm59.py` and `cylinder59.py` do with
  `commit()`.
- `scripts/preflight.py`: named by prompt 59; not in the tree at 809d1c6. The step adds it as
  CLAUDE.md's pre-commit checklist in one command (format, lint, the system map's `--check`,
  the suite's unit and integration tiers) or strikes the item.
- Stale branches: at 809d1c6, 17 local and 19 remote branches are merged into `main`
  (`git branch --merged main`, `git branch -r --merged main`); delete the merged ones, keeping
  the annotated tag `collocated-final` and the backup branch
  `backup/ecr-001-close-phase2-ecfa399`.

## J. Equipment vents (the amendment of 2026-10-16, step 0b)
**What the vents are.** Clean-room process tools carry vents of two kinds. Process exhaust is
ducted from the tool's body to the facility's exhaust: air leaves the room through an opening
on a tool face at a flow the facility's fan sets. Cooling discharge is room air drawn through
the tool's electronics and blown back into the room: a jet from a tool face, with an intake
elsewhere on the tool. Today a boundary segment can sit only on a domain edge, and every face
between a SOLID and a non-SOLID cell is a no-slip, no-flux wall, so neither kind can be
represented. Alex's decision of 2026-10-16: both, as airflow only (section H carries the heat
they omit). This section is design text for step 0b's own prompt; nothing of it is built in
step 0.

**The segment.** An obstacle-face segment is a boundary segment in the `boundaries` section,
so the event model (section B) replaces or sets it by name like any other (a tool's exhaust
failing, a cooling fan stopping), with `location: obstacle` and two keys of its own:
`obstacle`, the name of an obstacle in the `obstacles` list, and `face`, one of `top`,
`bottom`, `left`, `right`, the obstacle's face it sits on. Its extent along that face is
`x_start` and `x_end` for a top or bottom face and `y_start` and `y_end` for a side, as for a
domain-edge segment. Two existing types, no new one:
- `fixed_flow_outlet`, the process exhaust: the face's normal velocity is held outward from the
  room into the obstacle at `velocity`, which is required (a vent's flow is the facility's,
  stated, not a share of the remainder). It joins the stated outlets in ADR-012 decision 1's
  remainder rule, so the domain-edge returns that state no velocity share what the supply
  leaves after the hood and every exhaust vent, and the room stays balanced exactly on any mesh.
- `velocity_inlet`, the cooling discharge: the face's normal velocity is held into the room at
  `velocity`, normal to the face (`u_velocity` and `v_velocity` keep their domain-edge meaning
  for an angled jet), with `turbulence_intensity` and `dissipation_length` required with the
  turbulence section (ADR-012 C's inlet rule, as for the supply), and the concentration keys
  allowed as on any inlet. Its intake, when the tool draws from the room, is a second segment,
  a `fixed_flow_outlet` on another face of the same tool at the flow that balances it; a
  discharge with no intake adds its flow to the room, and the returns take it by the remainder
  rule.
`pressure_outlet` and `wall` are refused on an obstacle face: the zero-gradient copy needs the
face beyond the outlet, which is inside the obstacle, and the copy's weakness to entering air
(#61) is why the room has no pressure outlet; an obstacle face is a wall already.

**What each layer reads.**
- *Velocity layer* (`boundary_registry.py`, `boundary_staggered.py`). The registry resolves
  obstacle-face segments as it resolves edge segments, through the same coverage rule (first
  match in configuration order, wall by default), on the obstacle's face coordinates.
  `apply_normal_velocity` writes the prescribed normal velocity on the covered faces of the
  SOLID cells' boundary, which are interior entries of the face arrays, as it writes the
  domain's inlet faces, once before the loop; the inward-flux and the reference-flux sums
  (`get_total_inlet_flux`, the corrector's `flux_scale`) include the discharges, and the
  remainder rule's stated outflow includes the exhausts.
- *Momentum and pressure* (`momentum.py`, `pressure.py`). A face bounding a SOLID cell is not an
  unknown today and its value is read as zero (momentum.py's module notes, "Obstacles"). A
  vent face stays not an unknown, with its prescribed value read where zero is read now: the
  continuity of the fluid cell beside it (the corrector's right-hand side and
  `mass_imbalance`), and the QUICK boundary form at the obstacle, whose wall node takes the
  vent's normal velocity. The tangential component on a vent's span is the domain inlet's: the
  prescribed tangential velocity (zero for a normal jet) with the molecular viscosity, not the
  obstacle wall stencil. The p' equation keeps no coefficient across the face, as at a domain
  inlet. Configurations without vents are bitwise unchanged (the hashes of record, REQ-S16).
- *Wall treatment* (`turbulence.py`). `TurbulenceBoundary.wall_faces` drops the vent faces
  (a vent is not a wall: no wall function, no held eps from it, no wall production); a
  discharge's faces carry its inflow k and eps into the k and eps step, as a domain inlet's
  do; an exhaust's faces take the cell's k and eps out (upwind), as a domain fixed-flow outlet's
  do. A cell beside a vent and beside a wall elsewhere keeps the wall function on its wall
  faces only.
- *Concentration boundary* (`boundary_concentration.py`). A discharge's faces are inlet faces
  (`inflow_u`, `inflow_v` carry `concentration[k]`, HEPA-reduced when `hepa_filtered`, zero
  when absent); an exhaust's faces are outlet faces, with the outflow from the upwind cell and
  the budget booking it as outflow. Neither deposits: `surface_u` and `surface_v` are
  `SURFACE_NONE` on a vent's span, so the obstacle's deposition total counts its wall faces
  only.

**Amendment of 2026-10-16 (prompt 61, the build date): recirculated discharge.** A cooling fan
returns the room air its intake drew, contaminated as the room is; a zero default concentration
makes every fan a clean-air source and would bias operator exposure (M2) and removal (M3) in the
tool's favour, which is the comparison the product exists to make. A `velocity_inlet` vent takes
an optional `recirculate_from: <segment name>`, naming a `fixed_flow_outlet` vent on the same
obstacle. The discharge's concentration of each class is then the flux-weighted mean, over the
intake's faces, of the room cells each draws from, lagged one transport step: the mean is formed
from the field the step starts from, the field the explicit advection of that step reads.
`hepa_filtered` applies to it as to any inlet (the mean times one less the class's efficiency).
Refused at load: the named segment missing, not a fixed-flow vent, on another obstacle, or with
`concentration` also given. *Added 2026-10-16 (prompt 61b, review 61 S6):* a pair whose flows
differ is refused too, at load on the configured extents (to `BALANCE_TOLERANCE`) and on the
mesh on the faces each covers (to `MESH_BALANCE_TOLERANCE`), as the stated outlets' balance is: a
discharge of more flow than its intake would carry the intake's concentration at the larger
flow and create particles. The budget books the intake as outflow and the discharge as inflow,
so the pair nets to zero when the two flows are equal and nothing is filtered: at every step,
since both terms are formed from the same field, and so at steady state. Built in
`boundary_concentration.py` (`Recirculation`, `ConcentrationFaces.recirculation`) and
`solver_transport.py`; tested in `tests/test_solver_transport.py` (the pair's inflow equals its
outflow over forty steps of a non-uniform field, and a uniform field stays uniform under the pair
with no other source while a clean discharge empties the cells above it).

**Validated at load (REQ-C02).** `obstacle` names an obstacle that exists; `face` is one of the
four; the extent lies within the face's span and is not empty; the face does not lie on a domain
edge (an obstacle standing on the floor has no bottom face to vent from) or against another
obstacle; two vents on one face do not overlap; the type is `velocity_inlet` or
`fixed_flow_outlet`, a fixed-flow vent states its `velocity` and an inlet vent its
turbulence keys with the turbulence section; `obstacle` and `face` are refused on a segment
whose `location` is a domain edge, and required when it is `obstacle`; bools are
refused where numbers are expected and unknown keys refused, as for every other segment. On the
mesh, the extent must cover at least one face of the obstacle's boundary.

**The tests (step 0b).** In `tests/test_config.py`, one test per rejection path above. In the
layers' test files: the normal velocity written on exactly the covered obstacle faces and read
back from the corrected faces unchanged after a solve; the flux through a vent equals its
velocity times its span exactly; a closed box whose only openings are a discharge and an
exhaust of equal flow converges with the signed domain sum of the imbalance at rounding; the
remainder rule's returns share the supply less the hood and the exhausts; `wall_faces` excludes
the vent span and keeps the rest of the obstacle; a discharge carrying a concentration enters
the budget's inflow at velocity times span times concentration; an exhaust's outflow is booked
as outflow and no vent face deposits; and every configuration without vents bitwise unchanged
(the hashes of record). One product-grid measurement: the product room with a process exhaust
on a tool's side and a cooling discharge on a tool's top converges under the stopping rule.

**As built (step 0b, prompt 61, 2026-10-16): the departures from this section.** The section
was built as written except where these say otherwise, each with its reason; the build and its
measurements are `docs/reports/adr014_step0b.md`. *Departures 1, 3 and 4 ratified (orchestrator
and Alex, 2026-10-16, review 61 S4) as amendments of the accepted section.*
1. *No angled jet on a vent.* `u_velocity` and `v_velocity` are refused on a vent, where the
   section kept their domain-edge meaning. Three layers read the tangential velocity on an
   obstacle face as zero: the momentum wall stencil and its QUICK boundary form (the wall value
   at the face), and the k-epsilon corner gradients (a face beside a SOLID cell reads zero at
   the corner). Carrying a prescribed tangential value through each is a change to all three
   that no test of this section and no product use needs. A vent is a normal jet or a normal
   draw; an angled one is a later step's, with those three changes.
2. *The tangential viscosity on a vent's span is the domain inlet's as built, not the molecular
   one.* The section says the tangential component on a vent takes "the domain inlet's: the
   prescribed tangential velocity with the molecular viscosity". A domain inlet's tangential
   stencil takes the viscosity the stencil's face rule gives (`MomentumPredictor.
   stencil_viscosity`: air's on the laminar path, `mu + rho nu_t` of the wall cell in the
   coupled solve); only a wall face takes the wall function's. The vent takes exactly that: its
   span leaves `TurbulenceBoundary.wall_faces`, so the stencil keeps its face-rule viscosity,
   which is the molecular one on the laminar path. The section's first clause (the domain
   inlet's) is what was built; its "molecular" holds on the laminar path only.
3. *A closed box of stated flows loads.* The section's test names a closed box whose only
   openings are a discharge and an exhaust of equal flow, which `config.py` refused: with no
   pressure outlet it required a fixed-flow outlet that states no velocity. A file whose outlets
   all state their velocities now loads when the stated outflow equals the configured inflow to
   1e-9 relative, and `StaggeredBoundary` refuses it unless the faces balance to 1e-12 relative
   (`MESH_BALANCE_TOLERANCE`), since the mesh may round two openings differently. Unbalanced,
   both refusals stand as before.
4. *Two refusals the section did not list.* A vent's extent key on the other axis (`y_start` on
   a top vent) is refused, where a domain-edge segment ignores it: on a vent the slip would
   silently move nothing. An `obstacle` that two obstacles share is refused: obstacle names were
   never checked for uniqueness, and a vent must name one.
5. *What the section left unsaid, decided here.* A discharge's velocity enters
   `get_max_boundary_velocity`, the stopping rule's velocity scale, as an inlet's does (an
   exhaust's does not), and its faces count as inlet faces in the k-epsilon solve's initial
   state, weighted by their inflow. A vent face is first-order in the scalar scheme's
   neighbourhood: the face itself carries the inflow value in and the room cell's value out, as
   the section says, and the next face into the room reads its far-upstream node, inside the
   obstacle, as the upstream value (the scheme's rule beside any obstacle, unchanged), where a
   domain inlet's next face reads the inflow value as Leonard's boundary node.
6. *Where the vents are found.* `Mesh.obstacle_mask` is the one rule that places an obstacle
   (the classification reads it, cell for cell what it was), `staggered.obstacle_face_inputs`
   finds an obstacle face's room faces from it, the registry's `coverage_on` gives them to the
   vents and `boundary_staggered.vent_runs` is the derivation the velocity, turbulence and
   concentration layers all read (REQ-S12.1). `boundary_concentration` now imports
   `boundary_staggered` for it.

## Probes (committed before the runs; results after)

**Predictions (orchestrator, 2026-10-16).**
- (a) A warm-started re-solve of the product room to a changed boundary state takes under a
  third of a solve from rest.
- (b) One converged tier-2 time step of the cylinder at Re 100 costs under 50 outer iterations,
  and the showcase (a settled Strouhal number) fits in a few hours on this machine.

**Predictions (orchestrator, 2026-10-16, prompt 59b, on the clean-out probe of decision 2;
committed before it ran).**
- (c) The sensors' recovery times at Courant 0.4 differ from those at 0.1 by under 2%.
- (d) The volume average decays more slowly than the well-mixed exp(-N t) after the first air
  change, because the recirculation zones empty last.

**Builder (2026-10-16, before the runs, scored apart).**
- (a) Holds on outer count; the wall ratio is close to the outer ratio since both arms run the
  same room. The back arm (closed from open) costs about as much as the warm arm.
- (b) Holds at the 1e-4 U criterion by a wide margin; the long march's cost per step rises as
  shedding develops but stays under 50 outer iterations; one run of the paper's length at
  dt 0.05 fits in a few hours on the uniform grid with the probe's concurrency.
- (c) Holds for the 1e-2 crossings and most 1e-3 crossings; the 5 um class's late crossings in
  the lingering zones may differ by more, since the forward Euler error at 0.4 steepens
  gradients and the tail of a decay is where a small absolute error is a large relative one.
- (d) Holds: the supply short-circuits to the returns through the gaps, so the mean falls faster
  than well-mixed at first and slower later.

**Method.** `warm59.py` (T1), `cylinder59.py` (T2 to T4), `order59.py` (T5), `scales59.py` (T6),
`cleanout59.py` (T7, prompt 59b); each script's docstring states its arms and what it writes. Wall times were taken with three to
five processes sharing the machine and are stated with their process counts; counts are not
affected by concurrency.

**Results.** See `docs/reports/probe59/tables59.md` (T1 to T7), copied below.

### T1. The product room re-solved to the door's states (320x120)

| Arm | State solved | Start | Stop | Outer | Wall (s) | s per outer | Worst imbalance at end (kg/s per m) | Processes |
|---|---|---|---|---|---|---|---|---|
| L0 (step 8) | door closed | rest | error_estimate_and_continuity | 4,389 | 2151 | 0.490 | 6.52e-14 | 7 |
| rest | door open | rest | error_estimate_and_continuity | 4,579 | 881 | 0.192 | 1.21e-13 | 3 |
| warm | door open | L0, converged | error_estimate_and_continuity | 3,872 | 768 | 0.198 | 1.21e-13 | 3 |
| back | door closed | warm, converged | error_estimate_and_continuity | 3,447 | 680 | 0.197 | 1.42e-13 | 3 |

| Ratio | Outer | Wall |
|---|---|---|
| warm over rest (prediction (a): under one third) | 0.846 | 0.872 |
| back over rest | 0.753 | 0.772 |

### T2. The cylinder at Re 100: outer iterations per implicit Euler step, from the start-up field

| Arm | dt | Courant | pressure_rtol | Step tol (U) | Outer per step: mean, median, max | Mean outer to 1e-3, 1e-4, 1e-5, 1e-6 | s per step | s per outer | Imbalance (last step) | Processes |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.025 | 0.25 | 0.0001 | 0.0001 | 3.0, 3, 3 | 1.0, 3.0, -, - | 0.73 | 0.243 | 9.5e-10 | 5 |
| B | 0.05 | 0.50 | 0.0001 | 0.0001 | 4.0, 4, 4 | 2.0, 4.0, -, - | 1.03 | 0.257 | 5.0e-10 | 5 |
| C | 0.025 | 0.25 | 1e-08 | 0.0001 | 3.0, 3, 3 | 1.0, 3.0, -, - | 1.23 | 0.409 | 7.0e-14 | 5 |
| D | 0.05 | 0.50 | 0.0001 | 1e-05 | 27.8, 7, 212 | 2.0, 4.1, 27.8, - | 5.77 | 0.207 | 8.4e-10 | 5 |
| E | 0.05 | 0.50 | 0.0001 | 1e-06 | 162.9, 17.5, 500 | 2.0, 4.1, 21.4, 50.5 | 33.73 | 0.207 | 2.6e-10 | 5 |
| F | 0.025 | 0.25 | 0.0001 | 1e-06 | 199.6, 10, 500 | 1.2, 3.2, 19.2, 70.9 | 37.40 | 0.187 | 6.4e-10 | 5 |

Start-up, rerun after the arms (the field the arms began from was overwritten by a smoke run and is not kept; the steady loop is deterministic, so the rerun is the same field): 200 steady outer iterations from rest on 370x200 (74,000 cells), 0.258 s per outer with 3 processes.

### T3. The long marches from the perturbed start-up: the cost as shedding develops

Arm B_long: dt 0.05, step tolerance 0.0001 U, 800 steps (40 time units, 5.8 periods at St 0.146); wake sign changes over the march: 10; 5 processes.

| Steps | Time units | Mean outer per step | Max outer | s per step | Largest wake v | Sign changes in band | Time units between sign changes (twice is a period) |
|---|---|---|---|---|---|---|---|
| 1 to 200 | 0.1 to 10.0 | 7.0 | 7 | 1.63 | 0.1021 | 2 | 4.75 |
| 201 to 400 | 10.1 to 20.0 | 7.6 | 8 | 1.58 | 0.1612 | 3 | 3.85 |
| 401 to 600 | 20.1 to 30.0 | 8.0 | 8 | 1.61 | 0.2239 | 3 | 3.73 |
| 601 to 800 | 30.1 to 40.0 | 8.0 | 8 | 1.58 | 0.2466 | 2 | 3.80 |

Arm D_long: not run, or not completed.


### T4. The showcase's cost from T2 and T3: one run at Re 100 to Sohankar's length

Period 1 / St = 6.85 time units at St 0.146; the paper's run is 6300 steps of 0.025 = 157.5 time units, 23.0 periods. Hours = steps x seconds per step.

| Arm | dt | Steps per period | Steps for the paper's length | s per step (measured) | Hours, one run |
|---|---|---|---|---|---|
| A | 0.025 | 274 | 6300 | 0.73 | 1.28 |
| B | 0.05 | 137 | 3150 | 1.03 | 0.90 |
| C | 0.025 | 274 | 6300 | 1.23 | 2.15 |
| D | 0.05 | 137 | 3150 | 5.77 | 5.04 |
| E | 0.05 | 137 | 3150 | 33.73 | 29.51 |
| F | 0.025 | 274 | 6300 | 37.40 | 65.45 |
| B_long (last band) | 0.05 | 137 | 3150 | 1.58 | 1.38 |

### T5. The observed order of the transport scheme under refinement (VAL-008)

| Case | Scheme | Courant rule | Grids (dx, m) | Relative L2 error per grid | Peak retained per grid | Observed order between grids |
|---|---|---|---|---|---|---|
| aligned | umist | fixed | 0.04, 0.02, 0.01, 0.005 | 1.349e-01, 6.888e-02, 4.982e-02, 2.485e-02 | 0.838, 0.949, 0.986, 0.996 | 0.97, 0.47, 1.00 |
| aligned | umist | halving | 0.04, 0.02, 0.01, 0.005 | 1.349e-01, 6.008e-02, 2.712e-02, 9.420e-03 | 0.838, 0.924, 0.966, 0.984 | 1.17, 1.15, 1.53 |
| aligned | upwind | fixed | 0.04, 0.02, 0.01, 0.005 | 5.190e-01, 3.826e-01, 2.540e-01, 1.531e-01 | 0.470, 0.598, 0.726, 0.831 | 0.44, 0.59, 0.73 |
| aligned | upwind | halving | 0.04, 0.02, 0.01, 0.005 | 5.190e-01, 3.933e-01, 2.677e-01, 1.646e-01 | 0.470, 0.588, 0.713, 0.818 | 0.40, 0.55, 0.70 |
| front | umist | fixed | 0.04, 0.02, 0.01, 0.005 | 2.355e-02, 1.781e-02, 1.154e-02, 5.953e-03 | 1.000, 1.000, 1.000, 1.000 | 0.40, 0.63, 0.95 |
| front | umist | halving | 0.04, 0.02, 0.01, 0.005 | 2.355e-02, 1.080e-02, 4.142e-03, 1.303e-03 | 1.000, 1.000, 1.000, 1.000 | 1.12, 1.38, 1.67 |
| front | upwind | fixed | 0.04, 0.02, 0.01, 0.005 | 1.653e-01, 1.109e-01, 6.954e-02, 4.085e-02 | 0.992, 0.999, 1.000, 1.000 | 0.58, 0.67, 0.77 |
| front | upwind | halving | 0.04, 0.02, 0.01, 0.005 | 1.653e-01, 1.146e-01, 7.364e-02, 4.402e-02 | 0.992, 0.999, 1.000, 1.000 | 0.53, 0.64, 0.74 |
| front_wide | umist | fixed | 0.04, 0.02, 0.01, 0.005 | 9.674e-03, 5.507e-03, 2.771e-03, 1.376e-03 | 1.002, 1.001, 1.000, 1.000 | 0.81, 0.99, 1.01 |
| front_wide | umist | halving | 0.04, 0.02, 0.01, 0.005 | 9.674e-03, 3.097e-03, 8.699e-04, 2.332e-04 | 1.002, 1.001, 1.000, 1.000 | 1.64, 1.83, 1.90 |
| front_wide | upwind | fixed | 0.04, 0.02, 0.01, 0.005 | 7.436e-02, 4.194e-02, 2.255e-02, 1.175e-02 | 0.985, 0.995, 0.998, 0.999 | 0.83, 0.89, 0.94 |
| front_wide | upwind | halving | 0.04, 0.02, 0.01, 0.005 | 7.436e-02, 4.392e-02, 2.427e-02, 1.284e-02 | 0.985, 0.994, 0.998, 0.999 | 0.76, 0.86, 0.92 |
| oblique | umist | fixed | 0.04, 0.02, 0.01 | 1.934e-01, 7.465e-02, 4.128e-02 | 0.706, 0.885, 0.958 | 1.37, 0.85 |
| oblique | umist | halving | 0.04, 0.02, 0.01 | 1.934e-01, 7.349e-02, 3.060e-02 | 0.706, 0.865, 0.940 | 1.40, 1.26 |
| oblique | upwind | fixed | 0.04, 0.02, 0.01 | 6.318e-01, 4.678e-01, 3.092e-01 | 0.289, 0.441, 0.611 | 0.43, 0.60 |
| oblique | upwind | halving | 0.04, 0.02, 0.01 | 6.318e-01, 4.745e-01, 3.179e-01 | 0.289, 0.435, 0.602 | 0.41, 0.58 |
| stretched | umist | fixed | 0.04, 0.02, 0.01 | 2.279e-01, 1.028e-01, 6.436e-02 | 0.669, 0.822, 0.883 | 1.15, 0.68 |
| stretched | umist | halving | 0.04, 0.02, 0.01 | 2.279e-01, 1.078e-01, 6.584e-02 | 0.669, 0.809, 0.877 | 1.08, 0.71 |
| stretched | upwind | fixed | 0.04, 0.02, 0.01 | 6.530e-01, 5.182e-01, 4.105e-01 | 0.273, 0.391, 0.504 | 0.33, 0.34 |
| stretched | upwind | halving | 0.04, 0.02, 0.01 | 6.530e-01, 5.221e-01, 4.128e-01 | 0.273, 0.387, 0.501 | 0.32, 0.34 |

### T6. The product room's scales and the transport march's cost (320x120, L0 faces)

| Quantity | Value |
|---|---|
| Free area of the slice (m^2) | 17.52 |
| Supply flow (m^2/s per m depth) | 3.15 |
| Hood flow (m^2/s) | 0.45 |
| Return flow, four returns (m^2/s) | 2.7 |
| Return face velocity (m/s) | 1.227 |
| Air change rate N (1/s) | 0.1798 |
| Air changes per hour | 647.3 |
| Turnover 1 / N (s) | 5.562 |
| Well-mixed 100:1 recovery ln(100) / N (s) | 25.61 |
| Room length over supply speed (s) | 17.78 |
| Room height over supply speed (s) | 6.667 |
| Transport cfl_number | 0.1 |
| Wall seconds per simulated second, five classes | 92.67 |

| Class (um) | Settling velocity (m/s) | Height over settling (s) | Stable dt at cfl (s) | Steps per simulated s | Wall s per step | Wall s per simulated s |
|---|---|---|---|---|---|---|
| 0.1 | 8.790e-07 | 3.413e+06 | 0.0009058 | 1104 | 0.0209 | 23.08 |
| 0.3 | 4.271e-06 | 7.024e+05 | 0.0009058 | 1104 | 0.01742 | 19.24 |
| 0.5 | 1.007e-05 | 2.978e+05 | 0.0009058 | 1104 | 0.01591 | 17.56 |
| 1 | 3.517e-05 | 8.53e+04 | 0.0009058 | 1104 | 0.01537 | 16.97 |
| 5 | 7.779e-04 | 3857 | 0.0009058 | 1104 | 0.01433 | 15.82 |

### T7. The clean-out at Courant 0.1 and 0.4 (160x60, L0 flow, Sc_t 0.7): the time error

C0 = 1 in every non-SOLID cell at t = 0, the supply clean, the step the smaller stable_dt of the two classes (0.5 and 5 um), run until every sensor of both classes is below 1e-3 C0; N = 0.1798 per second (1 / N = 5.56 s).

| Arm | dt (s) | Steps | Simulated end (s) | Stopped by | Wall (s) | Wall s per simulated s | Budget relative residual | Processes |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 1.9781e-03 | 27,575 | 54.55 | sensors below 1e-3 | 380 | 6.97 | -1.2e-06, -1.2e-06 | 2 |
| 0.4 | 7.9124e-03 | 6,892 | 54.53 | sensors below 1e-3 | 98 | 1.79 | -4.0e-07, -4.0e-07 | 2 |

| Class (um) | Sensor | Level | First time below and staying below, Courant 0.1 (s) | Courant 0.4 (s) | 0.4 against 0.1 |
|---|---|---|---|---|---|
| 0.5 | near_door | 0.01 | 29.22 | 29.21 | -0.03% |
| 0.5 | near_door | 0.001 | 54.55 | 54.53 | -0.03% |
| 0.5 | above_gap_1 | 0.01 | 1.61 | 1.57 | -2.58% |
| 0.5 | above_gap_1 | 0.001 | 1.80 | 1.73 | -3.63% |
| 0.5 | above_gap_2 | 0.01 | 1.68 | 1.63 | -2.72% |
| 0.5 | above_gap_2 | 0.001 | 1.87 | 1.81 | -3.17% |
| 0.5 | hood_entry | 0.01 | 3.53 | 3.45 | -2.24% |
| 0.5 | hood_entry | 0.001 | 3.77 | 3.66 | -2.94% |
| 0.5 | free-area mean | 0.01 | not reached | not reached | - |
| 0.5 | free-area mean | 0.001 | not reached | not reached | - |
| 5 | near_door | 0.01 | 29.06 | 29.05 | -0.02% |
| 5 | near_door | 0.001 | 54.36 | 54.35 | -0.02% |
| 5 | above_gap_1 | 0.01 | 1.60 | 1.56 | -2.84% |
| 5 | above_gap_1 | 0.001 | 1.80 | 1.73 | -3.52% |
| 5 | above_gap_2 | 0.01 | 1.67 | 1.63 | -2.60% |
| 5 | above_gap_2 | 0.001 | 1.87 | 1.80 | -3.49% |
| 5 | hood_entry | 0.01 | 3.52 | 3.44 | -2.30% |
| 5 | hood_entry | 0.001 | 3.76 | 3.65 | -3.00% |
| 5 | free-area mean | 0.01 | not reached | not reached | - |
| 5 | free-area mean | 0.001 | not reached | not reached | - |

Largest relative difference of a sensor's recovery time, 0.4 against 0.1: 3.63% (prediction (c): under 2%).

| Air changes N t | t (s) | Well-mixed exp(-N t) | Free-area mean, 0.5 um, Courant 0.1 | 0.4 | Free-area mean, 5 um, Courant 0.1 | 0.4 |
|---|---|---|---|---|---|---|
| 0 | 0.0 | 1.000e+00 | 1.000e+00 | 1.000e+00 | 1.000e+00 | 1.000e+00 |
| 1 | 5.6 | 3.679e-01 | 1.929e-01 | 1.926e-01 | 1.922e-01 | 1.919e-01 |
| 2 | 11.1 | 1.353e-01 | 1.025e-01 | 1.024e-01 | 1.020e-01 | 1.020e-01 |
| 3 | 16.7 | 4.979e-02 | 7.103e-02 | 7.099e-02 | 7.063e-02 | 7.060e-02 |
| 4 | 22.2 | 1.832e-02 | 5.324e-02 | 5.322e-02 | 5.283e-02 | 5.281e-02 |
| 5 | 27.8 | 6.738e-03 | 4.212e-02 | 4.211e-02 | 4.170e-02 | 4.169e-02 |
| 6 | 33.4 | 2.479e-03 | 3.456e-02 | 3.455e-02 | 3.411e-02 | 3.410e-02 |
| 8 | 44.5 | 3.355e-04 | 2.507e-02 | 2.506e-02 | 2.458e-02 | 2.457e-02 |

**Reading the tables (written after the runs, 2026-10-16).**
- *T1, prediction (a): fails.* The warm arm's outer count is about 0.85 of the rest arm's and its
  wall time about the same fraction, not under a third. The two arms stop at the same state (the
  cell-centred velocities of `rest.npz` and `warm.npz` agree to better than 1e-9 m/s); the warm
  arm's first residual is about a third of the rest arm's, and the two residual tails are
  parallel about 700 outer iterations apart, the slow mode starting about 20 times smaller from
  the warm state; that offset is the saving. The back arm, the closed state from the open one, costs about three quarters of the
  rest arm and returns to the step 8 state L0 to better than 1e-9 m/s in every cell (`back.npz`
  against `results/builder56/L0_320x120.npz`): the closed room has one steady state on this grid
  whichever state the solve starts from. Decision 1's cache, not its warm start, is what saves a
  solve. The builder's prediction on (a) was wrong in the same direction as the orchestrator's.
- *T2, prediction (b): holds at the recommended criterion.* At 1e-4 U per step (arms A to C) a
  step converges in three or four outer iterations, every step, at either pressure tolerance and
  either time step; the 1e-8 pressure tolerance costs about 1.7 times the wall per outer of 1e-4
  for the same count. At 1e-5 U (arm D) most steps take seven, and a slow mode stalls some for
  tens to a couple of hundred; at 1e-6 U (arms E and F) a quarter to a third of the steps reach
  the 500 cap without meeting it. The per-step count is the criterion's, and 1e-4 U is where the
  step's cost is the typical step's rather than the stalls'.
- *T3 and T4.* Arm B's long march from the perturbed start keeps 7 to 8 outer iterations per step
  at 1e-4 U over 40 time units while the wake's oscillation grows band by band; the sign changes
  of the wake velocity come about 3.8 time units apart in the last three bands, a period of
  about 7.6 and a Strouhal number of about 0.13, 10% below the paper's 0.146 and twice the 5%
  threshold decision 11 proposes, read off a growing oscillation on the uniform probe grid with
  freestream lateral boundaries; VAL-022's clustered grid is meant to close that gap, and
  decision 11's options (3) and (4) are the next changes if it does not. One run of the paper's length at arm B's step costs
  about an hour and a half at the measured seconds per step with five processes sharing the
  machine (T4); the 1e-5 U march (arm D's tag `_long`, `logs/cylinder_long.log`) was stopped at
  step 576 of 2,400 with its per-step count at 26 to 44 in the log's samples (one every 48
  steps) as the oscillation grew, and wrote no record. Prediction (b): its per-step half holds on the probe grid; its showcase half, the
  time to a settled Strouhal number, is not measured: T3 is still growing at 40 time units (the
  largest wake velocity rises band by band), the paper's 157.5 time units stand in for the
  settling time in T4, and the proposed clustered grid's cost per step is not measured, so that
  half is scored not measured, not holds.
- *T5, VAL-008.* At a fixed Courant number every case's order tends to 1 under refinement, the
  limited scheme and upwind alike: forward Euler's. With the Courant number halved per
  refinement the Gaussian pulse's order under the limited scheme lies between 1 and 2 on the
  aligned and oblique channels, lower on the stretched mesh, and upwind's is below 1 at these
  resolutions (its asymptotic 1 is not reached while the pulse is five to twenty cells wide). The
  smooth monotone front's order under the halving rule rises grid by grid toward 2 as the front
  gains cells: the narrow front (2.5, 5, 10 and 20 cells across its width on the four grids)
  does not reach 1.8 on any pair, and the wide front (5, 10, 20 and 40 cells) is within 0.2 of 2
  on its two finest pairs, whose coarser grids hold 10 and 20 cells across the width. At the
  same cell counts the two fronts differ (1.67 against 1.83 at 10 to 20 cells): the narrow one
  travels 10 widths to the wide one's 3, and the limiter's departure from QUICK where the
  gradient ratio leaves 1 at the front's shoulders, first order and shrinking with the spacing,
  accumulates with the travel. Decision 10's row is therefore the wide front as measured, with
  the Courant number halved, its order read between the two finest grids; two pairs stand
  behind the rule.
- *T6.* The transport march at the file's Courant number costs about 93 wall seconds per
  simulated second for the five classes on the product grid, about 1,100 steps per simulated
  second per class; the step is set by the return faces. The well-mixed 100:1 time of the slice
  is about 26 s against a turnover of about 5.6 s.
- *T7, predictions (c) and (d) (prompt 59b).* (c) fails, narrowly: the largest relative
  difference of a sensor's recovery time between 0.4 and 0.1 is 3.6%, at the three sensors under
  the supply, which clear within two seconds; the near-door sensor, which clears last, differs
  by 0.03%, and the free-area mean's curve is the same to three figures. Every 0.4 time is
  earlier than its 0.1 time, the forward Euler steepening's direction. The builder's (c), which
  expected the late 5 um crossings to differ most, was wrong about where the difference sits:
  it is in the fast crossings, where a few hundredths of a second are a few percent. (d) holds,
  more strongly than predicted: the mean falls below the well-mixed line in the first air
  change (the supply short-circuits to the returns through the gaps) and then far above it,
  about two orders of magnitude at eight air changes, as the recirculation zones empty last; the
  mean's own 100:1 time is not reached by the time every sensor is below 1e-3. The clean-out on
  this room is a sensor-placement question, which section C's recovery field is there to
  answer.

## Consequences
- The transport march, not the flow, sets a scenario's cost (T6); decision 2 and Phase 6's GPU
  port are where it goes. The flow re-solve per event is T1's warm arm.
- `solve_steady` gains a warm start and `StaggeredSolver` a transient loop; `momentum.py` a
  transient term; `mesh.py` interior clustering; `config.py` the scenario schema with inheritance.
  Each is a cascade row when built, and each is bitwise the current path when its argument is
  absent.
- The scenario files are four configurations extending the default, not four rooms.
- Every run leaves one HDF5 file that can be continued; a capped steady solve can be continued
  from its checkpoint, which prompts 49 and 52 could not.
- VAL-008 has a quantity, a case and a Courant rule; the cylinder has an identifier, three rows
  and OPEN thresholds; both go into PROJECT_PLAN when decided.
- The door's physics is an assumption with its sources stated; the exchange velocity is a key,
  and a measurement would replace it without a code change.

## Alternatives considered
A single transient solver for both tiers (every room scenario run time-accurately): T6's cost
per simulated second would be the transport's plus a flow solve per step; decision 1 of
2026-10-16 ruled it out for turbulent rooms in 2D. Ramped events: section A. Recirculation of
room air through the supply: ADR-011 decision 3, no recirculation in v1, stands. Scenario
events that change obstacles (a tool moved): a layout change, which `tests/test_layouts.py`'s
files already express, not an event.

## Sources
1. Sohankar, A., Norberg, C. and Davidson, L. (1998), "Low-Reynolds-number flow around a square
   cylinder at incidence: study of blockage, onset of vortex shedding and outlet boundary
   condition", Int. J. Numer. Meth. Fluids 26, 39-56. Fetched 2026-10-16 from
   https://www.cfd-sweden.se/lada/postscript_files/Sohankar_num-fluids.pdf (the PDF's text
   extracted with pdftotext): sections 2 and 2.1 (SIMPLEC, Crank-Nicolson, QUICK, dt 0.025,
   the residual criterion 0.001, d = 0.004, Nb = 20, the far-field D, Xu = 10, symmetry
   lateral boundaries), section 4.1 and Table II (NBC against CBC, Xd, iterations per step
   Nt and Np, 6,300 steps at Re 100), Table IV (the refinement study, St 0.146 at Re 100),
   section 4.4 and Table V (Re_cr 51.2 at zero incidence, +/- 1).
2. ISO 14644-3, Test methods, Annex B, the recovery test (B.12 in the 2005 edition; B.4 in the
   2019 edition, not fetched: the standard is paywalled). Its definition as the secondary sources
   below state it: the 100:1 recovery time from a challenge at 100 times the target level, the
   alternative evaluation by recovery rate from a straight-line segment of the semi-log decay
   over at least five successive measurement pairs, no acceptance limit set by the standard,
   intended for non-unidirectional airflow rooms in the as-built or at-rest state. Beckman
   Coulter, "What is the ISO 14644-3 cleanroom recovery test?",
   https://www.beckman.com/resources/industry-standards/iso-14644/iso-14644-3 (search result,
   2026-10-16).
3. Cleanroom Technology, "What is the difference between clean-up period and recovery time?",
   https://cleanroomtechnology.com/what-is-the-difference-between-clean-up-period (fetched
   2026-10-16): the 100:1 method, 1:10 for ISO 8 at rest, the target not exceeding 1.5 times
   the baseline, "no normative requirement for a specific recovery time"; EU GMP Annex 1 (2008)
   clean-up period of 15 to 20 minutes as a guidance value, and the 2023 Annex 1 clause 4.29
   note (iii) with "less than 20 minutes".
4. ASHRAE, "Cleanroom pressurization" (bookstore excerpt, ashrae-d-da-13-011.pdf, search result
   2026-10-16): ISO 14644-4 allows a pressure differential of 5 to 20 Pa between adjacent areas,
   the value left to the designer.
5. Bhattacharya, A., Metcalf, A. R., Nafchi, A. M. and Mousavi, E. S. (2021), "Particle
   dispersion in a cleanroom: effects of pressurization, door opening and traffic flow",
   Building Research and Information 49(3), 294-307, DOI 10.1080/09613218.2020.1720500 (search
   result, 2026-10-16; the full text was not fetched): door operation can eliminate and even
   reverse the pressurisation across the door; higher differentials keep aerosols out better and
   recover positive pressure sooner.
6. Kalliomaki, P., Saarinen, P., Tang, J. W. and Koskela, H. (2015), "Airflow patterns through
   single hinged and sliding doors in hospital isolation rooms", International Journal of
   Ventilation (search result, 2026-10-16, via the AIVC abstract): hinged-door exchange volumes
   of 1.2 to 2.4 m^3 per opening cycle in still air, a passage adding about 0.4 m^3.
7. Shu, C.-W. and Osher, S. (1988), "Efficient implementation of essentially non-oscillatory
   shock-capturing schemes", J. Comput. Phys. 77, 439-471: the SSP Runge-Kutta family, Heun's
   method as its two-stage member (cited through ADR-011 C; not fetched).
8. Issa, R. I. (1986), "Solution of the implicitly discretised fluid flow equations by
   operator-splitting", J. Comput. Phys. 62, 40-65: PISO (not fetched; decision 4's option (2)).
9. The project's records: `docs/reports/ecr002_step8_product.md` (L0 on 320x120, the sources,
   the L2 cycle), `docs/reports/ecr002_step6_comparative.md` section 5.4 (the marches at
   Courant 0.4 checked against 0.1), `docs/reports/phase3_transport_report.md` (the limits),
   `results/builder56/L0_320x120.{json,npz}` (the warm start's source state), ADR-011 B, C and H,
   ADR-012 decision 1 and C, ADR-013 B, GitHub issue 53 (D5, D6), issues 61 and 69, tests 55 and
   56 (`docs/prompts/test-55.md`, `test-56.md`).
