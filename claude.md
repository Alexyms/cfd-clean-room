# CFD Clean Room Simulation -- Claude Code Project Configuration

## Project Overview

Computational Fluid Dynamics engine built from scratch in Python (with compiled C inner loops) simulating clean room airflow and contamination transport. Finite Volume discretization, incompressible laminar Navier-Stokes, Eulerian multi-class particle transport with gravitational settling. Alert monitoring layer for contamination detection and sensor placement analysis.

**This is a portfolio project. Code quality, readability, and professional engineering practices matter as much as functionality.**

---

## Language & Runtime

- **Primary language:** Python 3.12+
- **Performance-critical code:** C (compiled to shared library, called via ctypes)
- **Virtual environment:** `.venv/` in project root. Always activate before running or installing anything.
- **Package management:** pip with `requirements.txt` for runtime dependencies, `requirements-dev.txt` for dev/test dependencies.

---

## Project Structure

```
cfd_clean_room/
├── src/                    # All Python source modules
├── csolver/                # C source files, header, Makefile
├── configs/                # YAML configuration files
├── tests/                  # Validation and unit tests
├── scripts/                # Simulation runners, visualization
├── results/                # Simulation output (gitignored)
├── docs/                   # Architecture docs, ADRs, system doc
├── .github/workflows/      # CI/CD and code review automation
├── claude.md               # This file
├── pyproject.toml          # Ruff config, project metadata
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Dev dependencies (ruff, pytest, etc.)
└── README.md
```

---

## Python Coding Standards

### Formatting

Use `ruff` for all formatting and linting. Run before every commit:

```bash
ruff format .
ruff check . --fix
```

If `ruff check` reports errors that `--fix` cannot resolve, fix them manually before committing.

### Configuration

Ruff configuration lives in `pyproject.toml`. Do not override ruff settings with inline comments (`# noqa`) unless there is a documented reason in the comment itself. Example of acceptable suppression:

```python
# noqa: E501 -- equation readability is more important than line length here
long_equation = (rho_p * d_p**2 * g * cunningham_correction) / (18 * mu)
```

### Type Hints

Full type hints on all functions, methods, and class attributes. No exceptions.

```python
# Yes
def settling_velocity(self, size_class: int) -> float:

# Yes -- complex types use the modern syntax
def solve_timestep(
    self,
    C_k: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    size_class: int,
    dt: float,
    v_ext: np.ndarray | None = None,
) -> np.ndarray:

# No -- missing return type
def settling_velocity(self, size_class: int):

# No -- using Any as a cop-out
def process_results(self, data: Any) -> Any:
```

Use `np.ndarray` for numpy arrays. If a function accepts or returns arrays with specific shapes, document the expected shape in the docstring, not the type hint.

### Docstrings -- NumPy Style

Every public class, method, and function gets a docstring. Private helpers (`_name`) get a docstring if the logic is non-obvious.

```python
def settling_velocity(self, size_class: int) -> float:
    """Compute Stokes settling velocity with Cunningham slip correction.

    Uses the Stokes drag law corrected for slip at small particle sizes.
    Valid for Re_p << 1 (Stokes regime), which holds for all particle
    sizes in the clean room operating range.

    Parameters
    ----------
    size_class : int
        Index into the configured particle sizes array (0-4).

    Returns
    -------
    float
        Terminal settling velocity in m/s. Positive downward.

    Notes
    -----
    v_s = (rho_p * d_p^2 * g * C_c) / (18 * mu)

    where C_c is the Cunningham slip correction factor.
    """
```

For classes, the docstring goes on the class, not on `__init__`:

```python
class ParticlePhysics:
    """Compute size-dependent particle transport properties.

    Provides settling velocity, Brownian diffusion coefficient,
    Cunningham slip correction, and deposition velocity for each
    configured particle size class.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration containing particle sizes,
        density, and fluid properties.
    """

    def __init__(self, config: SimConfig) -> None:
        ...
```

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Modules | `snake_case` | `solver_ns.py`, `time_integration.py` |
| Classes | `PascalCase` | `NavierStokesSolver`, `ParticlePhysics` |
| Functions/methods | `snake_case` | `compute_residual`, `apply_velocity_bc` |
| Constants | `UPPER_SNAKE_CASE` | `BOLTZMANN_CONSTANT`, `GRAVITY` |
| Private methods | `_leading_underscore` | `_pressure_correction_step` |
| Local variables | `snake_case` | `cell_type`, `settling_vel` |
| Physics variables | Short names acceptable when standard | `u`, `v`, `p`, `rho`, `mu`, `dt`, `dx`, `dy` |
| Size class index | `k` or `size_class` | -- |
| Array indices | `i`, `j` for grid; `n` for time | -- |

Physics shorthand (`u`, `v`, `p`, `rho`, `mu`, `C_k`, `dx`, `dt`) is acceptable and preferred in solver code where it matches standard notation. Always define the variable in the docstring of the function where it first appears.

### Imports

Group imports in this order, separated by blank lines:

1. Standard library
2. Third-party packages
3. Project modules

```python
import ctypes
from pathlib import Path

import numpy as np
import yaml

from src.config import SimConfig
from src.mesh import Mesh
```

Never use wildcard imports (`from module import *`).

---

## Writing Style

Applies to all human-readable text: docstrings, comments, commit messages, documentation, README, ADRs, and PR descriptions.

### Avoid AI Writing Patterns

This project will be read by engineers evaluating the author's ability. Text that reads like AI-generated output undermines that goal. Specifically:

**Do not use:**
- Em dashes. Use commas, periods, parentheses, or rewrite the sentence.
- Unicode symbols in prose (arrows, bullet characters, check marks, etc.). Use plain ASCII in comments and docs.
- "Leverage", "utilize", "facilitate", "streamline", "robust", "comprehensive", "cutting-edge", "state-of-the-art", "delve into", "it's worth noting that", "importantly"
- Filler hedging: "It should be noted that", "As mentioned previously", "In order to"
- Rhetorical questions as transitions
- Exclamation points in technical writing
- Sycophantic openings: "Great question!", "That's a really interesting point"

**Do use:**
- Short, direct sentences.
- "Use" instead of "utilize". "Build" instead of "architect" (as a verb). "Show" instead of "demonstrate".
- Hyphens (-) for compound modifiers (e.g., "size-dependent") and in CLI flags. Double hyphens (--) for long-form flags only.
- Plain language. If a simpler word works, use it.

### Comments

Comments explain *why*, not *what*. The code shows what happens. The comment explains the reasoning.

```python
# Yes -- explains a non-obvious choice
# Cunningham correction matters below 1 um where continuum assumptions break down.
# Above 1 um the correction is ~1.0 and has no practical effect.
correction = cunningham_slip_correction(d_p, mean_free_path)

# No -- restates the code
# Calculate the Cunningham slip correction
correction = cunningham_slip_correction(d_p, mean_free_path)
```

### Docstrings

Docstrings describe the contract: what the function accepts, what it returns, and any assumptions or limitations. Keep the one-line summary genuinely brief. Put physics context in the Notes section, not the summary.

```python
# Yes
def settling_velocity(self, size_class: int) -> float:
    """Compute Stokes settling velocity with Cunningham slip correction.

    Parameters
    ----------
    ...
    """

# No -- summary is a paragraph
def settling_velocity(self, size_class: int) -> float:
    """This function computes the terminal settling velocity of a
    particle in the Stokes regime, taking into account the Cunningham
    slip correction factor which accounts for non-continuum effects
    at small particle sizes.

    Parameters
    ----------
    ...
    """
```

---

## C Coding Standards

### Files

- All C source lives in `csolver/`.
- Shared header `csolver.h` defines all function signatures exposed to Python.
- `Makefile` compiles to a shared library (`libcsolver.so` on Linux, `libcsolver.dylib` on macOS).

### Style

- Function names: `snake_case`, prefixed with module context. Example: `ns_pressure_correction_step`, `transport_advection_diffusion_step`.
- No dynamic memory allocation inside hot loops. All arrays allocated in Python and passed as pointers.
- Every function documents its parameters with a comment block above the signature.
- Compile with `-Wall -Wextra -O2`. Fix all warnings.

### Interface Contract

C functions receive raw pointers and dimensions. They have no knowledge of:
- Boundary conditions (applied in Python before/after C call)
- Configuration (parameters passed as individual arguments)
- File I/O, logging, or error reporting to the user

```c
/*
 * Perform one pressure correction step (SIMPLE algorithm).
 *
 * u, v:       velocity field arrays [ny x nx], row-major
 * p:          pressure field array [ny x nx], row-major
 * nx, ny:     grid dimensions
 * dx, dy:     cell spacing
 * dt:         time step
 * rho, mu:    fluid density and viscosity
 * alpha_p:    pressure under-relaxation factor
 *
 * Modifies u, v, p in-place.
 */
void ns_pressure_correction_step(
    double *u, double *v, double *p,
    int nx, int ny,
    double dx, double dy, double dt,
    double rho, double mu,
    double alpha_p
);
```

---

## Configuration

### Single Source of Truth

All simulation parameters live in YAML configuration files under `configs/`. No hardcoded physical constants, grid sizes, solver parameters, or threshold values in source code.

Physical constants (Boltzmann constant, gravity, standard air properties) are defined once in a dedicated constants section of the config or as module-level constants in a single `constants.py` file. They are never duplicated across modules.

If you find yourself typing a number that isn't 0, 1, or 2 into solver code, it should probably be in the config.

### Config Validation

`config.py` validates all parameters at load time. Missing keys, out-of-range values, and type mismatches raise clear errors immediately, not when the solver hits them 10 minutes into a run.

---

## Testing

### Framework

Use `pytest`. All test files in `tests/`. Test file naming: `test_<module>.py`.

Tests are organized into three tiers. All three tiers must pass before a PR is merge-ready.

### Tier 1: Unit Tests

Every module gets unit tests that verify individual functions and classes in isolation. These test correctness of logic, error handling, edge cases, and interface compliance. They do not require a running simulation or interaction between modules.

Examples:
- `config.py`: rejects missing keys, out-of-range values, wrong types, malformed YAML
- `mesh.py`: correct cell counts, cell classification, neighbor lookup at boundaries and corners, degenerate grid dimensions
- `particles.py`: returns correct types, handles edge-case diameters, zero-division guards
- `boundary.py`: raises on unrecognized boundary types, applies BCs to correct cell faces
- `monitor.py`: alert fires at exact threshold, no false positives below threshold, handles empty sensor list
- `scenarios.py`: event activation at correct times, event expiration, overlapping events

Mark unit tests with `@pytest.mark.unit`:

```python
@pytest.mark.unit
def test_config_rejects_negative_viscosity() -> None:
    """SimConfig raises ValueError when fluid viscosity is negative."""
```

### Tier 2: Integration Tests

Integration tests verify that modules interact correctly. Data flows through multiple modules in the correct format and produces reasonable outputs. These catch interface mismatches, dtype errors, and shape incompatibilities that unit tests miss.

Examples:
- Config loads and initializes Mesh without errors
- Mesh produces arrays that the NS solver accepts (correct shape, dtype, memory layout)
- NS solver output feeds into transport solver without shape or type errors
- Transport solver output feeds into alert monitor correctly
- Full timestep loop (NS solve, transport solve for all size classes, monitor update) runs without error
- C solver produces identical results to a pure NumPy reference implementation

Mark integration tests with `@pytest.mark.integration`:

```python
@pytest.mark.integration
def test_ns_output_feeds_transport_solver() -> None:
    """NS solver velocity fields are accepted by transport solver without error."""
```

### Tier 3: Validation Tests

Validation tests verify physical correctness against analytical solutions and published benchmarks. These are the phase gates from the development plan (VAL-001 through VAL-009). They assert quantitative pass/fail criteria as defined in the Validation Test Matrix.

Mark validation tests with `@pytest.mark.validation` and reference the VAL ID:

```python
@pytest.mark.validation
def test_poiseuille_flow_val001() -> None:
    """VAL-001: Poiseuille flow -- L2 error < 1% vs analytical parabolic profile."""
```

### pytest Configuration

Configure markers in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests -- isolated function and class behavior",
    "integration: Integration tests -- module interaction and data flow",
    "validation: Validation tests -- physics verification against analytical solutions",
]
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run by tier
pytest tests/ -v -m unit
pytest tests/ -v -m integration
pytest tests/ -v -m validation

# Run a specific validation test
pytest tests/test_poiseuille.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Phase Completion Reports

Each development phase concludes with a structured report generated in `docs/reports/`. Reports are partially automated (test results, coverage metrics) and partially written (scope changes, implementation decisions, lessons learned). The report serves as the phase gate evidence package.

### Report Location and Naming

```
docs/reports/
    phase1_foundation_report.md
    phase2_navier_stokes_report.md
    phase3_transport_report.md
    phase4_scenarios_report.md
    phase5_alert_system_report.md
    phase6_visualization_report.md
```

### Report Template

Each report follows this structure:

```markdown
# Phase N Completion Report: [Phase Name]

**Date:** YYYY-MM-DD
**Phase Gate Verdict:** PASS / FAIL
**Branches Merged:** list of PRs merged in this phase

## Scope

### Planned
Reference the development plan. List planned deliverables and validation gates.

### Delivered
What was actually built. Note any additions or removals from the original plan.

### Deferred
Anything planned but moved to a later phase, with rationale.

## Test Results

### Unit Tests
- Tests run: N
- Passed: N
- Failed: N
- Skipped: N

### Integration Tests
- Tests run: N
- Passed: N
- Failed: N

### Validation Tests
| ID      | Description          | Criterion         | Result | Value   |
|---------|----------------------|-------------------|--------|---------|
| VAL-00X | [name]               | [pass criterion]  | PASS   | [actual]|

### Coverage
- Line coverage: N%
- Branch coverage: N%
- Uncovered modules or functions: list any gaps and why they exist

## Implementation Decisions

Document any decisions made during implementation that were not
anticipated in the development plan. Each entry follows the ADR format:
what was decided, what alternatives existed, why this choice was made.

If a decision warrants a formal ADR, add it to docs/ADR/ and reference
it here.

## Deviations from Architecture

List any changes to module interfaces, data flow, or configuration
schema that differ from the development plan. For each deviation:
what changed, why, and whether SYSTEM.md was updated to reflect it.

## Lessons Learned

Brief notes on what went well and what was harder than expected.
These inform planning for subsequent phases.
```

### Automated Report Data

The CI pipeline collects the following automatically on phase completion:
- pytest results in JUnit XML format (`pytest --junitxml=reports/results.xml`)
- Coverage report (`pytest --cov=src --cov-report=json:reports/coverage.json`)
- List of merged PRs and commit hashes for the phase

The human-written sections (scope changes, implementation decisions, lessons learned) are filled in manually before the report is committed. The report itself is committed to `main` as the final action of the phase, with commit message `docs: add phase N completion report`.

### Phase Gate Criteria

A phase passes its gate when:
1. All planned validation tests for that phase pass
2. All unit and integration tests pass (not just the new ones)
3. No ruff errors
4. Coverage does not decrease from the previous phase
5. SYSTEM.md is current with any interface or architecture changes
6. The completion report is filled out and committed

---

## Version Control

### Branch Naming

Format: `<category>/<short-description>`

| Category | Use |
|----------|-----|
| `phase1/`, `phase2/`, etc. | Development work within a specific phase |
| `feature/` | Cross-phase features or enhancements |
| `fix/` | Bug fixes |
| `docs/` | Documentation-only changes |
| `ci/` | CI/CD and automation changes |
| `refactor/` | Code restructuring without behavior change |

Examples:
- `phase1/particles-module`
- `phase1/mesh-generation`
- `phase2/ns-solver-simple`
- `fix/boundary-corner-cells`
- `ci/code-review-action`
- `docs/adr-update-turbulence`

### Commit Messages -- Conventional Commits

Format: `type: concise description`

| Type | Use |
|------|-----|
| `feat:` | New functionality |
| `fix:` | Bug fix |
| `test:` | Adding or modifying tests |
| `docs:` | Documentation changes |
| `refactor:` | Code restructuring, no behavior change |
| `chore:` | Project maintenance, dependency updates |
| `build:` | Build system, compilation, Makefile changes |
| `ci:` | CI/CD workflow changes |

Rules:
- Lowercase type prefix, colon, space, then description.
- Description starts with a verb: "add", "fix", "implement", "update", "remove".
- No period at the end.
- Keep the first line under 72 characters.
- Reference the module or component: `feat: implement Cunningham correction in particles module`
- Reference validation IDs when relevant: `test: add Poiseuille flow validation VAL-001`

### Pull Requests

All code reaches `main` through pull requests. No direct commits to `main`.

PR title follows the same Conventional Commits format as the primary commit.

PR description should include:
- What changed and why
- Which phase/validation case this relates to
- Any open questions or known limitations

The automated review system will check every PR against the architecture docs, coding standards, and development plan.

### What Gets Committed

**Always committed:**
- Source code, tests, configs, documentation
- `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`
- CI workflow files
- C source files, header, Makefile

**Never committed (add to `.gitignore`):**
- `.venv/`
- `results/` (simulation output)
- `__pycache__/`, `*.pyc`
- `*.so`, `*.dylib`, `*.o` (compiled artifacts -- built locally or in CI)
- `.DS_Store`
- IDE-specific files (`.vscode/` settings, but `.vscode/extensions.json` is ok)

---

## Architecture Rules

These rules enforce the design decisions documented in the Architecture Decision Records (ADRs) in `docs/`. The code review system checks compliance with these rules.

### Centralized Configuration

No simulation parameter may be defined in more than one location. All parameters flow from YAML config through `SimConfig` to consuming modules. If a module needs a value, it receives it from config or from a parent module that received it from config.

**Violation example:** A solver module defines `rho = 1.2` as a local default.
**Correct:** The solver receives `config.rho` at initialization.

### Interface Contracts

Module boundaries are defined by the interface contracts in the development plan. Changes to a module's public interface (adding/removing/changing function signatures) require:
1. Updating the interface contract in `docs/SYSTEM.md`
2. Updating all calling modules
3. Noting the change in the PR description

### Extension Point Preservation

The transport equation's external force field interface (`v_ext` parameter) must remain in the solver even though it is unused in v1. Do not remove it to "clean up" unused parameters. It is an architectural decision (ADR-007), not dead code.

### Separation of Concerns

- C code handles computation only. No I/O, no config parsing, no boundary logic.
- Python handles orchestration: config loading, boundary conditions, time stepping coordination, output, monitoring.
- The alert monitor reads concentration fields but never modifies them.
- Scenarios modify boundary conditions and source terms but never touch the solver internals.

### No Cross-Module State Mutation

Modules do not reach into other modules' internal state. Data flows through function arguments and return values. If module A needs data from module B, B provides it through a public method, not by exposing internal arrays.

---

## Pre-Commit Checklist

Before committing, verify:

1. `ruff format .` produces no changes
2. `ruff check .` reports no errors
3. `pytest tests/ -v` passes all existing tests
4. No hardcoded parameters in solver code
5. All new functions have type hints and docstrings
6. Commit message follows Conventional Commits format
7. Branch name follows the naming convention
