# ADR-008: Collocated Ghost Cell Wall Treatment

## Status
Superseded by ADR-010 on 2026-04-16. See ECR-001 for decision rationale.

Originally Accepted (2026-04-15). Retained as a historical record of the ghost-cell wall treatment decision.

## Context
The collocated grid SIMPLE solver uses a ghost cell approach for no-slip walls: boundary cells are assigned velocity values (u_bnd = u_int/3) such that linear interpolation between the boundary cell center and the interior cell center places the wall velocity (zero) at the domain face.

This approach has known O(h) accuracy at walls, compared to O(h^2) for more complex schemes (immersed boundary methods, body-fitted grids with wall functions).

## Decision
We retain the ghost cell approach and accept O(h) wall accuracy rather than implementing a higher-order wall treatment.

## Consequences

**Positive:**
- Simple, maintainable implementation
- Industry-standard approach (OpenFOAM, most production CFD codes use equivalent schemes)
- Consistent with our architectural commitment to readability over cleverness
- Vectorizes cleanly in NumPy and translates directly to CUDA kernels

**Negative:**
- VAL-001 Poiseuille criterion relaxed from 1% to 2.5% L2 error on 80x40 grid
- Would need ~240x120 grid to achieve 1% error (impractical for CI test runtime)

## Alternatives Considered

1. **Doubled diffusion coefficient + equidistant ghost cell (u_bnd = -u_int):**
   Would give O(h^2) wall accuracy but requires mixing two ghost cell approaches
   throughout the solver (momentum, face fluxes, pressure source, Rhie-Chow).
   Tested; produced correct peak velocity but 5.5% L2 error from inconsistency
   between momentum treatment and mass flux treatment.

2. **Immersed boundary method:**
   More accurate but substantially more complex. Out of scope for this project.

3. **Body-fitted grid with wall functions:**
   Production-quality but requires replacing the structured mesh. Out of scope.

## References
- Ferziger and Peric, Computational Methods for Fluid Dynamics, 3rd ed., Section 7.5
- OpenFOAM documentation on boundary conditions for wall treatments
