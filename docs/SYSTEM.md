# System Architecture Document

**Project:** CFD Clean Room Simulation
**Status:** Phase 1 in progress. Foundation modules under development.
**Last Updated:** 2026-04-15

This document is the single reference for system architecture, requirements, module interfaces, and dependency relationships. The automated code review system reads this document on every PR to verify compliance. Keep it current.

---

## 1. System Description

A from-scratch Computational Fluid Dynamics engine simulating clean room airflow and contamination transport. The system solves incompressible Navier-Stokes equations for a 2D velocity field using the Finite Volume method, then solves advection-diffusion equations for particle concentration across five size classes on top of that velocity field. An alert monitoring layer tracks contamination against ISO 14644 thresholds for reactive detection and proactive sensor placement analysis.

The simulation domain is a vertical cross-section of a semiconductor clean room with HEPA supply vents, return vents, an entry door, process equipment, and a laminar flow hood.

---

## 2. Requirements

Requirements are organized by subsystem. Each requirement has a unique ID, a rationale, and a traceability link to the validation test or architectural rule that verifies it.

### 2.1 Solver Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-S01 | The NS solver shall converge to a steady-state velocity field with residuals below a configurable tolerance. | Velocity field accuracy depends on convergence. Divergent or under-converged solutions produce meaningless transport results. | VAL-001, VAL-002 |
| REQ-S02 | The NS solver shall reproduce the Poiseuille flow parabolic velocity profile with L2 error < 1%. | Validates basic FV discretization and pressure-velocity coupling against an exact analytical solution. | VAL-001 |
| REQ-S03 | The NS solver shall reproduce lid-driven cavity centerline velocity profiles within 2% of Ghia et al. (1982) benchmark data. | Validates nonlinear advection, 2D pressure gradients, and recirculation handling. | VAL-002 |
| REQ-S04 | The velocity field shall satisfy the incompressibility constraint (divergence-free) to within configurable tolerance at every cell. | Mass conservation is fundamental. FV enforces this by construction, but numerical errors can accumulate. | VAL-007 |
| REQ-S05 | The solver shall use the SIMPLE algorithm for pressure-velocity coupling. | Industry-standard approach. Well-documented, stable, compatible with structured grids. | Architecture review |
| REQ-S06 | The pressure correction inner loop shall be implemented in compiled C, called from Python via ctypes. | Compute-bound operation requires compiled performance. Python handles orchestration. | Integration test |

### 2.2 Transport Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-T01 | The transport solver shall solve the advection-diffusion equation for particle concentration on the velocity field produced by the NS solver. | Core physics coupling. Particles are carried by airflow and spread by diffusion. | VAL-003, VAL-004 |
| REQ-T02 | The transport solver shall support five discrete particle size classes: 0.1, 0.3, 0.5, 1.0, and 5.0 um. | Spans the relevant physics regimes from diffusion-dominated to settling-dominated and maps to ISO 14644 classification sizes. | Unit test |
| REQ-T03 | Each size class shall have independent settling velocity computed via Stokes drag with Cunningham slip correction. | Settling velocity varies by 1000x across the size range. Cunningham correction is significant below 1 um. | VAL-005 |
| REQ-T04 | Each size class shall have independent Brownian diffusion coefficient computed via Stokes-Einstein relation with Cunningham correction. | Diffusion dominates transport for sub-micron particles. | VAL-006 |
| REQ-T05 | The transport solver shall conserve total particle mass to within 0.01% across all timesteps (mass in domain + mass out = mass in + mass from sources). | FV formulation guarantees conservation by construction. This test catches numerical bugs. | VAL-007 |
| REQ-T06 | The transport solver shall accept an external force field parameter (v_ext) for future extension to electrostatic precipitation modeling. The parameter shall be structurally present but set to zero in v1. | Architectural extensibility per ADR-007. Avoids future refactoring of the solver interface. | Architecture review |
| REQ-T07 | Pure diffusion from a point source shall produce a Gaussian concentration profile with L2 error < 1% vs the analytical solution. | Validates the diffusion discretization independently of advection. | VAL-003 |
| REQ-T08 | Advection of a concentration pulse in a uniform flow shall preserve peak location to within 1 cell width and maintain pulse shape. | Validates advection discretization. Excessive numerical diffusion indicates the scheme is too dissipative. | VAL-004 |
| REQ-T09 | The particles module shall compute gravitational and diffusional deposition velocity for each size class, parameterized by boundary layer thickness and surface orientation (floor, ceiling, wall). | Deposition velocity is a boundary condition input for the transport solver. Floor deposition includes gravitational settling; ceiling and wall deposition are diffusion-only. | VAL-010 |
| REQ-T10 | The particles module shall estimate HEPA filter collection efficiency for each size class via interpolation of reference efficiency data. | HEPA efficiency determines the particle removal rate at supply vent boundaries. Efficiency varies by particle size with a minimum at the most-penetrating particle size (~0.3 um). | VAL-011 |

### 2.3 Configuration Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-C01 | All simulation parameters shall be defined in a single YAML configuration file. No simulation parameter shall be hardcoded in source modules. | Lesson from Stock Transformer: scattered parameters cause synchronization failures. | Architecture review |
| REQ-C02 | The configuration loader shall validate all parameters at load time and raise clear errors for missing keys, out-of-range values, and type mismatches. | Fail fast. Do not let invalid config propagate to a solver crash 10 minutes into a run. | Unit test |
| REQ-C03 | Scenario configuration files shall inherit from a base configuration and override specific parameters only. | Avoids duplicating the full config for each scenario. Changes to base config automatically propagate. | Unit test |
| REQ-C04 | Physical constants (Boltzmann constant, gravity, air properties at standard conditions) shall be defined in exactly one location. | No duplication of constants across modules. | Architecture review |

### 2.4 Alert System Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-A01 | The alert monitor shall detect threshold exceedance within one timestep of occurrence. | Detection latency of zero or one timestep. Delayed detection defeats the purpose of monitoring. | VAL-009 |
| REQ-A02 | Sensor locations shall be configurable via the YAML configuration. | Enables comparison of sensor placement strategies without code changes. | Unit test |
| REQ-A03 | Contamination thresholds shall be configurable per particle size class, following ISO 14644-1 classification limits. | Different size classes have different regulatory limits. Class 5 allows zero 5.0 um particles but 3.52 million 0.5 um particles per cubic meter. | Unit test |
| REQ-A04 | The monitor shall report detection latency per sensor per scenario: elapsed time from event onset to first threshold exceedance at each sensor. | Core metric for reactive monitoring. Answers "how fast do we know about a contamination event?" | Integration test |
| REQ-A05 | The monitor shall support evaluation of multiple sensor configurations across multiple scenarios to identify optimal placement. | Core value of proactive design mode. Answers "where should sensors go?" | Integration test |
| REQ-A06 | The alert monitor shall read concentration fields but never modify them. | Separation of concerns. Monitoring is observation, not intervention. | Architecture review |

### 2.5 Visualization Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-V01 | The system shall produce animated visualizations of contamination evolution over time for each scenario. | Primary portfolio deliverable. Visual demonstration of physics and system behavior. | Manual review |
| REQ-V02 | Visualizations shall support per-size-class display showing how the same event produces different spatial patterns for different particle sizes. | Demonstrates the physics of multi-class transport, not just pretty pictures. | Manual review |
| REQ-V03 | Visualizations shall overlay alert sensor locations with status indicators that change state when thresholds are exceeded. | Connects the physics simulation to the monitoring system visually. | Manual review |

### 2.6 Numerical Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-N01 | The simulation shall enforce the CFL condition for numerical stability at every timestep. | Explicit advection schemes are conditionally stable. Violating CFL produces divergent solutions. | Unit test |
| REQ-N02 | The solution shall demonstrate grid convergence at the expected order of accuracy when the grid is refined. | Confirms numerical accuracy. Second-order scheme should reduce error by 4x when grid spacing is halved. | VAL-008 |
| REQ-N03 | The C solver inner loop shall produce results identical to a pure NumPy reference implementation for all validation cases. | Verifies the Python-C interface is not introducing bugs through memory layout, dtype, or indexing errors. | Integration test |

---

## 3. Module Dependency Map

This section defines which modules depend on which. When a PR modifies a module, the code reviewer checks that downstream modules are still compatible.

### 3.1 Dependency Graph

```
config.py
    |
    +--> mesh.py
    |       |
    |       +--> boundary.py
    |       |       |
    |       |       +--> solver_ns.py
    |       |       |       |
    |       |       |       +--> solver_transport.py
    |       |       |               |
    |       |       |               +--> time_integration.py
    |       |       |                       |
    |       |       |                       +--> io_manager.py
    |       |       |
    |       |       +--> solver_transport.py (also depends on boundary)
    |       |
    |       +--> solver_ns.py (also depends on mesh directly)
    |       +--> solver_transport.py (also depends on mesh directly)
    |       +--> monitor.py (reads mesh for coordinate mapping)
    |
    +--> particles.py
    |       |
    |       +--> solver_transport.py (uses settling velocity, diffusion coeff)
    |
    +--> scenarios.py
    |       |
    |       +--> time_integration.py (queries active sources each timestep)
    |       +--> boundary.py (modifies BCs during events)
    |
    +--> monitor.py
            |
            +--> time_integration.py (updated each timestep)

csolver/ (C shared library)
    |
    +--> solver_ns.py (calls pressure correction via ctypes)
    +--> solver_transport.py (calls advection-diffusion via ctypes)
```

### 3.2 Cascade Rules

When a PR modifies a module, the reviewer verifies impact on downstream modules. Read this table as: "If you change X, check Y."

| Modified Module | Check These Downstream Modules | What to Check |
|-----------------|-------------------------------|---------------|
| config.py | mesh, boundary, solver_ns, solver_transport, particles, monitor, scenarios, time_integration | New/changed/removed config fields are handled by all consumers. No module accesses a field that no longer exists. No module ignores a new field it should use. |
| mesh.py | boundary, solver_ns, solver_transport, monitor | Grid dimensions, cell arrays, and coordinate arrays are consumed correctly. Shape assumptions still hold. |
| boundary.py | solver_ns, solver_transport | BC application interface unchanged. New BC types handled in solvers if needed. |
| solver_ns.py | solver_transport, time_integration | Velocity/pressure field output shape, dtype, and semantics unchanged. |
| solver_transport.py | time_integration, monitor | Concentration field output shape, dtype, and semantics unchanged. |
| particles.py | solver_transport | Settling velocity, diffusion coefficient interface unchanged. Return types and units unchanged. |
| scenarios.py | time_integration, boundary | Source term and BC modification interfaces unchanged. Event timing semantics unchanged. |
| monitor.py | time_integration | Update and query interfaces unchanged. Alert output format unchanged. |
| csolver/ | solver_ns, solver_transport | C function signatures match ctypes declarations. Memory layout assumptions (row-major, contiguous, double precision) unchanged. |
| time_integration.py | io_manager | Timestep sequence and output trigger logic unchanged. |

### 3.3 Cross-Cutting Concerns

These concerns span multiple modules. Changes to any of them require checking all modules that participate.

**Array conventions:** All 2D field arrays are shape `[ny, nx]`, row-major, contiguous, dtype `float64`. Every module that creates, passes, or receives a field array must follow this convention. A change to array layout or dtype cascades to every module.

**Unit system:** SI throughout. Meters, seconds, kilograms, Pascals. No CGS, no imperial, no implicit unit conversions. All values in config are SI.

**Coordinate system:** Origin at bottom-left of the domain. x increases left-to-right, y increases bottom-to-top. Gravity acts in the -y direction. Consistent across mesh, boundary, solver, and visualization.

---

## 4. Interface Contracts (Summary)

Detailed interface contracts are in the development plan. This section provides a quick reference for the reviewer.

### config.py --> all modules

```
SimConfig:
    room_width, room_height: float (meters)
    nx, ny: int
    rho, mu: float (SI)
    particle_sizes: list[float] (meters)
    particle_density: float (kg/m^3)
    dt, t_end: float (seconds)
    output_interval: int
    boundaries: dict[str, BoundarySpec]
    scenarios: list[ScenarioSpec]
    sensors: list[SensorSpec]
    thresholds: dict[str, float]
```

### mesh.py --> solver_ns, solver_transport, boundary, monitor

```
Mesh:
    x, y: ndarray (face coordinates)
    xc, yc: ndarray (cell center coordinates)
    dx, dy: float
    cell_type: ndarray[ny, nx] (FLUID=0, SOLID=1, BOUNDARY=2)
```

### particles.py --> solver_transport

Note: the constructor accepts raw parameters as an interim design.
It will be refactored to accept SimConfig when config.py is implemented.

```
ParticlePhysics:
    __init__(particle_sizes, particle_density, temperature, mu, mean_free_path, boundary_layer_thickness=1e-3)
    settling_velocity(size_class: int) -> float
    diffusion_coeff(size_class: int) -> float
    cunningham_correction(size_class: int) -> float
    deposition_velocity(size_class: int, surface: str) -> float
    hepa_efficiency(size_class: int) -> float
```

### solver_ns.py --> solver_transport, time_integration

```
NavierStokesSolver:
    solve_steady() -> tuple[ndarray, ndarray, ndarray]  # u, v, p
    solve_timestep(u, v, p, dt) -> tuple[ndarray, ndarray, ndarray]
    All output arrays: shape [ny, nx], dtype float64, contiguous
```

### solver_transport.py --> time_integration, monitor

```
TransportSolver:
    solve_timestep(C_k, u, v, size_class, dt, v_ext=None) -> ndarray
    Output array: shape [ny, nx], dtype float64, contiguous
```

### monitor.py --> time_integration

```
AlertMonitor:
    update(C_fields: dict[int, ndarray], t: float) -> None
    get_alerts() -> list[Alert]
    get_detection_latency(event: str) -> dict[str, float]
```

### scenarios.py --> time_integration, boundary

```
ScenarioManager:
    get_active_sources(t: float) -> list[SourceTerm]
    get_bc_modifications(t: float) -> dict
```

---

## 5. Scope Boundaries

### In Scope (v1)

- 2D vertical cross-section domain
- Incompressible laminar Navier-Stokes (SIMPLE algorithm)
- Finite Volume on structured rectangular grid
- Staircase representation of internal obstacles
- Five-class Eulerian particle transport with settling and diffusion
- Cunningham slip correction for sub-micron particles
- Three contamination scenarios: door seal leak, HEPA filter breach, equipment dust release
- Alert monitoring with configurable sensors and ISO 14644 thresholds
- Sensor placement comparison across scenarios
- Animated visualizations per size class
- Hybrid Python/C implementation
- Centralized YAML configuration
- Phase-gated development with validation tests before code

### Out of Scope (v1)

- 3D simulation (architecture supports extension, not implemented)
- Turbulence modeling (laminar assumption is physically justified per ADR-004)
- Thermal buoyancy / hot equipment convection
- Ionizer / electrostatic precipitation (extension point preserved per ADR-007)
- Unstructured meshes
- Multi-room or multi-bay simulation
- Real-time sensor hardware integration
- GPU acceleration (architecture supports extension via structured grid)
- Particle-particle interactions, coagulation, electrostatic effects
- Thermophoresis

### Scope Change Process

Any change to scope boundaries during development must be:
1. Documented in the PR description with rationale
2. Reflected in this document
3. Reflected in docs/PROJECT_PLAN.md
4. Reviewed for cascade impact using the dependency map above

---

## 6. Architecture Decision Records

Full ADRs are in the development plan document. Summary reference:

| ADR | Decision | Key Rationale |
|-----|----------|---------------|
| ADR-001 | Finite Volume discretization | Industry standard (Fluent, OpenFOAM). Conservation built into the method. |
| ADR-002 | 2D vertical cross-section | Captures gravity, HVAC flow, stratification. 3D adds complexity without proportional insight. |
| ADR-003 | Structured grid, staircase boundaries | Clean room geometry is rectangular. Staircase is exact for axis-aligned obstacles. |
| ADR-004 | Laminar flow assumption | Clean rooms are engineered for laminar flow. Re well below transition. Physically correct, not a shortcut. |
| ADR-005 | Hybrid Python/C | Python for orchestration, C for compute-bound inner loops. Mirrors production CFD architecture. |
| ADR-006 | Five particle size classes | Spans diffusion-dominated to settling-dominated regimes. Maps to ISO 14644. |
| ADR-007 | Ionizer modeling deferred | Scope risk. Extension point (v_ext) preserved in transport solver interface. |

---

## Document History

| Date | Change | Author |
|------|--------|--------|
| 2026-04-14 | Initial version. Architecture defined pre-development. | Alex Moroz-Smietana |
