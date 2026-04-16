# CFD Clean Room Simulation

A from-scratch Computational Fluid Dynamics engine simulating clean room airflow and contamination transport. Built in Python with a NumPy reference solver for validation and CUDA C++ acceleration via pybind11 for production runs.

The simulation models a vertical cross-section of a semiconductor clean room, solving incompressible Navier-Stokes for the velocity field using the Finite Volume method, then solving advection-diffusion equations for particle contamination transport across five size classes. An alert monitoring layer tracks contamination against ISO 14644 thresholds for detection and sensor placement analysis.

**Status:** Phase 2 in progress. Foundation modules (config, mesh, particles) complete and validated. Navier-Stokes solver under development.

## Architecture

- **Solver:** Finite Volume discretization, SIMPLE algorithm for pressure-velocity coupling
- **Transport:** Eulerian multi-class particle transport with gravitational settling and Cunningham slip correction
- **Particle sizes:** 0.1, 0.3, 0.5, 1.0, 5.0 um (spanning diffusion-dominated to settling-dominated regimes)
- **Implementation:** Python orchestration with NumPy reference solver; CUDA C++ acceleration via pybind11 for production runs
- **Alert system:** Configurable sensor placement, ISO 14644-1 threshold monitoring, detection latency analysis

## Development Approach

This project is planned before built. Architecture decisions, requirements, validation test cases, and module interfaces are defined before solver code is written. Development follows a phase-gated process with validation at each stage. An automated code review system (Claude Sonnet + Opus advisor) checks every PR against the architecture docs and coding standards.

See `docs/` for the system architecture, project plan, and development phasing.

## Validation

Each phase passes a validation gate before the next begins. Test cases include:

- Poiseuille flow (analytical parabolic velocity profile)
- Lid-driven cavity (Ghia et al. 1982 benchmark)
- Pure diffusion (analytical Gaussian spreading)
- Pulse advection (transport without distortion)
- Stokes settling velocity with Cunningham correction
- Mass conservation across all timesteps
- Grid convergence at expected order of accuracy

## Project Context

Third project in a portfolio series demonstrating systems engineering across domains:

1. **Stock Transformer** -- ML system with MBSE documentation (quantitative finance)
2. **NYC Taxi Network** -- Network analysis with data visualization (urban infrastructure)
3. **CFD Clean Room** -- Physics simulation with alert system (semiconductor manufacturing)

Portfolio: [alexyms.github.io](https://alexyms.github.io)

## Setup

```bash
git clone https://github.com/Alexyms/cfd-clean-room.git
cd cfd-clean-room
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## License

TBD
