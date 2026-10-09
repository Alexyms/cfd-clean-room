"""Momentum predictor on the staggered grid with QUICK advection by deferred correction.

ECR-001 step 4 (REQ-S07, REQ-S09). Given the current u, v and p and the
boundary data of src/boundary_staggered.py, form the discrete steady momentum
equations at the interior u and v faces, apply one under-relaxed Jacobi
sweep, and return the predicted velocities u* and v* with the diagonal
coefficients the pressure correction of step 5 needs. Nothing here
integrates into solve_steady and nothing corrects pressure.

Control volumes. ``u[j, i]`` sits at ``(x[i], yc[j])``; its control volume
spans ``xc[i-1]..xc[i]`` by ``y[j]..y[j+1]``. ``v[j, i]`` sits at
``(xc[i], y[j])``; its control volume spans ``x[i]..x[i+1]`` by
``yc[j-1]..yc[j]``. The mass flux through a streamwise face is the average
of the two stored velocities bounding it times the face length; through a
transverse face it is the sum over the two half cells of the stored normal
velocity times the half width. With these the net mass flux over a momentum
control volume is half the sum of the continuity residuals of the two cells
it straddles, so a divergence-free field gives zero net mass flux on every
momentum control volume and a uniform field is advected without change.

Deferred correction. QUICK's three-node stencil carries a negative
coefficient, which breaks the diagonal dominance Jacobi relies on. The
implicit matrix is therefore assembled with first-order upwind, whose
neighbour coefficients are never negative, and the difference between the
QUICK and the upwind advective flux, evaluated on the current field, is
carried as an explicit source. At convergence the source closes the gap and
the solution is the QUICK solution; only the path to it uses upwind
(Ferziger and Peric ch. 5, Versteeg and Malalasekera ch. 5). The diagonal
follows the collocated convention ``a_P = sum(a_nb) + (F_e - F_w + F_n -
F_s)``, so it equals the neighbour sum wherever the mass fluxes satisfy
continuity and exceeds it under under-relaxation.

QUICK on this mesh. A face value is the quadratic through the two nodes
bounding the face and the next node upstream, evaluated at the face
coordinate with Lagrange weights formed from the actual node positions.
Stretching is therefore exact and on a uniform mesh the weights reduce to
6/8, 3/8 and -1/8 (Leonard 1979). Where the far-upstream node lies outside
the domain the boundary form of Leonard's appendix is used: the quadratic
is fitted through the boundary value at its physical location and the two
nearest interior nodes. For the tangential component beside a wall this
gives, on a uniform mesh, ``phi_f = phi_C + (phi_D - phi_wall) / 3``; for a
streamwise face beside an inflow boundary, ``3/8 phi_B + 6/8 phi_1 - 1/8
phi_2``. A boundary that carries no value (a pressure outlet, zero
gradient) contributes a node holding the adjacent interior value.

Boundary entries. The boundary columns of u and rows of v are read as given
and never written: Dirichlet faces were set by StaggeredBoundary and outlet
faces are the caller's to extrapolate. The tangential wall values and wall
distances come from ``tangential_conditions``; the wall shear on the first
interior node is ``(phi_P - phi_wall) / wall_distance`` times the face
width, with no value stored outside the domain.

Obstacles. A u or v face bounding a SOLID cell is not an unknown and is
predicted as zero. A neighbouring unknown sees it as a fixed zero at its
storage location; the half-cell wall distance applies at domain edges only.
Obstacle accuracy is not a Phase 2 validation target.

A viscosity field (ADR-012 D, ECR-002 step 4). ``predict`` takes an optional
dynamic viscosity per cell, ``mu_eff``; without it the scalar lines run
unchanged. A component's streamwise faces lie at cell centres and take the
cell's value. Its transverse faces lie at cell corners: across the row
boundary in each of the two columns the face spans, the distance-weighted
harmonic mean of the two cells (the flux crosses them in series), then the
width-weighted arithmetic mean of the two columns (the half faces carry it
in parallel). The two domain-edge rows take the width-weighted mean of the
two wall cells. For the harmonic mean a SOLID cell takes the value of the
non-SOLID cell across the row boundary from it, so an obstacle face carries
the wall cell's viscosity as a domain edge does; no SOLID value is read.
Both means are written so that equal inputs return the input exactly, so a
uniform field equal to air's viscosity gives the scalar conductances to the
bit. With a varying viscosity the viscous force on u carries
``d/dx(mu du/dx) + d/dy(mu dv/dx)`` beyond ``div(mu grad u)``. On the MAC grid
that pair is ``mu d/dx(div u)``, which the scalar path drops as zero on a
divergence-free field, plus a part from the variation of mu. Form b
(docs/reports/ecr002_step0_frozen_viscosity.md, appendix B) drops the first
for the field too and keeps the second: at each face, the sum over the four
faces of its control volume of ``(mu_face - mu_f)`` times that face's stress
flux, ``mu_f`` the width-weighted mean of the face's two cells. It is exactly
zero for any uniform field, and it is carried with the deferred correction
as an explicit source on the current field. This is the probe's arithmetic
(``frozen34.py``'s ``FrozenPredictor``) written as committed code.

The wall viscosity (ADR-012 B). ``wall_mu`` replaces the viscosity of the
wall stencil face by face, for the wall functions of ECR-002 step 6. Without
it the stencil takes air's viscosity or the field's edge value.

Momentum sweeps. ``solver.momentum_sweeps`` Jacobi sweeps of the
under-relaxed equations run per call, on the coefficients and sources of the
current field, with the relaxation term held at the outer iterate. One
sweep is the committed sweep.
"""

from dataclasses import dataclass

import numpy as np

from src.boundary_staggered import StaggeredBoundary, TangentialCondition
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.staggered import u_shape, v_shape


@dataclass(frozen=True)
class MomentumCoefficients:
    """Discrete momentum equation of one velocity component, in its own shape.

    At every unknown face ``a_P phi_P = a_s_plus phi_(s+) + a_s_minus
    phi_(s-) + a_t_plus phi_(t+) + a_t_minus phi_(t-) + b_boundary +
    b_deferred + b_pressure``, where ``s`` is the component's own axis (x
    for u, y for v) and ``t`` the other. For u, ``s+`` is east, ``s-``
    west, ``t+`` north and ``t-`` south; for v, ``s+`` is north and ``t+``
    east. Every array has the component's staggered shape and is zero at
    faces that are not unknowns (boundary faces and faces of SOLID cells).
    The transverse coefficient toward a domain edge is zero, the wall's
    contribution sitting in ``a_P`` and ``b_boundary`` instead.

    Parameters
    ----------
    a_p : np.ndarray
        Diagonal, not under-relaxed. Positive exactly at the unknowns.
    a_s_plus, a_s_minus, a_t_plus, a_t_minus : np.ndarray
        Neighbour coefficients, all non-negative.
    b_boundary : np.ndarray
        Known Dirichlet contributions from the transverse domain edges.
    b_deferred : np.ndarray
        The QUICK minus upwind advective flux of the current field, moved
        to the right-hand side, plus form b's stress source when a viscosity
        field is given. Zero for a uniform velocity field.
    """

    a_p: np.ndarray
    a_s_plus: np.ndarray
    a_s_minus: np.ndarray
    a_t_plus: np.ndarray
    a_t_minus: np.ndarray
    b_boundary: np.ndarray
    b_deferred: np.ndarray


@dataclass(frozen=True)
class MomentumPrediction:
    """What the pressure correction of ECR-001 step 5 receives from the predictor.

    Parameters
    ----------
    u_star : np.ndarray
        Predicted x-velocity, shape [ny, nx+1]. Boundary columns are the
        input values; faces of SOLID cells are zero.
    v_star : np.ndarray
        Predicted y-velocity, shape [ny+1, nx], likewise.
    a_p_u : np.ndarray
        Diagonal momentum coefficient of every u face, shape [ny, nx+1],
        before under-relaxation, as the collocated solver's d-coefficient
        uses it. Positive exactly at the unknown faces and zero at
        boundary and SOLID faces, so ``a_p_u > 0`` is the mask of faces the
        pressure correction may correct and ``d = A_face / a_p_u`` is
        defined precisely there.
    a_p_v : np.ndarray
        The same for v, shape [ny+1, nx].
    """

    u_star: np.ndarray
    v_star: np.ndarray
    a_p_u: np.ndarray
    a_p_v: np.ndarray


@dataclass(frozen=True)
class _Orientation:
    """Mesh arrays for one component with its own axis last.

    ``s`` is the component's own axis (nodes at faces ``s_faces``), ``t``
    the transverse axis (nodes at centers ``t_centers``). ``low`` and
    ``high`` are the tangential conditions on the two transverse edges and
    ``solid`` is the SOLID mask with shape [nt, ns].
    """

    s_faces: np.ndarray
    s_centers: np.ndarray
    ds_cell: np.ndarray
    ds_face: np.ndarray
    t_faces: np.ndarray
    t_centers: np.ndarray
    dt_cell: np.ndarray
    dt_face: np.ndarray
    low: TangentialCondition
    high: TangentialCondition
    solid: np.ndarray


def _lagrange_weights(
    x_a: np.ndarray, x_b: np.ndarray, x_c: np.ndarray, x_f: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights of the quadratic through three nodes, evaluated at x_f."""
    w_a = (x_f - x_b) * (x_f - x_c) / ((x_a - x_b) * (x_a - x_c))
    w_b = (x_f - x_a) * (x_f - x_c) / ((x_b - x_a) * (x_b - x_c))
    w_c = (x_f - x_a) * (x_f - x_b) / ((x_c - x_a) * (x_c - x_b))
    return w_a, w_b, w_c


def _lerp(a: np.ndarray, b: np.ndarray, frac_b: np.ndarray) -> np.ndarray:
    """Weighted arithmetic mean ``a + (b - a) frac_b``; exactly ``a`` when ``a == b``."""
    return a + (b - a) * frac_b


def _harmonic(a: np.ndarray, b: np.ndarray, frac_a: np.ndarray) -> np.ndarray:
    """Weighted harmonic mean ``1 / (frac_a / a + (1 - frac_a) / b)``.

    Written as ``a`` plus a correction so that it returns ``a`` exactly when
    ``a == b``; both must be positive.
    """
    frac_b = 1.0 - frac_a
    return a + (b - a) * (a * frac_b) / (frac_a * b + frac_b * a)


def quick_face_values(
    phi: np.ndarray,
    nodes: np.ndarray,
    left: np.ndarray,
    faces: np.ndarray,
    positive: np.ndarray,
) -> np.ndarray:
    """QUICK face values along the last axis by quadratic interpolation.

    Parameters
    ----------
    phi : np.ndarray
        Node values, shape [..., N].
    nodes : np.ndarray
        Node coordinates, shape [N], increasing, N >= 3.
    left : np.ndarray
        For each face, the index of the node on its low side, shape [M].
        The face lies between nodes ``left`` and ``left + 1``.
    faces : np.ndarray
        Face coordinates, shape [M].
    positive : np.ndarray
        Shape [..., M], True where the flow crosses the face toward
        increasing coordinate.

    Returns
    -------
    np.ndarray
        Face values, shape [..., M].

    Notes
    -----
    The quadratic passes through the upstream node C, the downstream node
    D and the next node upstream of C. Where that node does not exist the
    next node downstream of D is used instead, which is Leonard's boundary
    form once the caller has placed the boundary value at its physical
    location as the end node.
    """
    n = nodes.shape[0]
    if n < 3:
        raise ValueError("QUICK needs at least three nodes along the axis")
    k = np.asarray(left)
    far_pos = np.where(k - 1 >= 0, k - 1, k + 2)
    far_neg = np.where(k + 2 <= n - 1, k + 2, k - 1)
    w_pos = _lagrange_weights(nodes[far_pos], nodes[k], nodes[k + 1], faces)
    w_neg = _lagrange_weights(nodes[far_neg], nodes[k + 1], nodes[k], faces)
    f_pos = (
        w_pos[0] * phi[..., far_pos]
        + w_pos[1] * phi[..., k]
        + w_pos[2] * phi[..., k + 1]
    )
    f_neg = (
        w_neg[0] * phi[..., far_neg]
        + w_neg[1] * phi[..., k + 1]
        + w_neg[2] * phi[..., k]
    )
    return np.where(positive, f_pos, f_neg)


class MomentumPredictor:
    """Predict u* and v* on the staggered grid with QUICK advection.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh; at least two cells on each axis.
    config : SimConfig
        Supplies rho, mu, alpha_velocity and momentum_sweeps.
    boundary : StaggeredBoundary
        Supplies the tangential wall values and wall distances.
    """

    def __init__(
        self, mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary
    ) -> None:
        ny, nx = mesh.cell_type.shape
        if nx < 2 or ny < 2:
            raise ValueError("the momentum predictor needs at least 2 cells per axis")
        self._u_shape = u_shape(mesh)
        self._v_shape = v_shape(mesh)
        self._rho = config.rho
        self._mu = config.mu
        self._alpha = config.alpha_velocity
        self._n_sweeps = config.momentum_sweeps
        tangential = boundary.tangential_conditions()
        solid = mesh.cell_type == SOLID
        self._live = ~solid
        self._for_u = _Orientation(
            s_faces=mesh.x,
            s_centers=mesh.xc,
            ds_cell=mesh.dx_cell,
            ds_face=mesh.dx_face,
            t_faces=mesh.y,
            t_centers=mesh.yc,
            dt_cell=mesh.dy_cell,
            dt_face=mesh.dy_face,
            low=tangential["bottom"],
            high=tangential["top"],
            solid=solid,
        )
        self._for_v = _Orientation(
            s_faces=mesh.y,
            s_centers=mesh.yc,
            ds_cell=mesh.dy_cell,
            ds_face=mesh.dy_face,
            t_faces=mesh.x,
            t_centers=mesh.xc,
            dt_cell=mesh.dx_cell,
            dt_face=mesh.dx_face,
            low=tangential["left"],
            high=tangential["right"],
            solid=np.ascontiguousarray(solid.T),
        )
        # The faces whose wall viscosity reaches an unknown's equation, in
        # the global [ny+1, nx+1] corner layout of wall_mu.
        self._wall_read = {
            "u": self._wall_faces(self._for_u),
            "v": np.ascontiguousarray(self._wall_faces(self._for_v).T),
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def momentum_coefficients(
        self, u: np.ndarray, v: np.ndarray
    ) -> tuple[MomentumCoefficients, MomentumCoefficients]:
        """Assemble the u and v momentum equations on the current field.

        Parameters
        ----------
        u : np.ndarray
            x-velocity, shape [ny, nx+1].
        v : np.ndarray
            y-velocity, shape [ny+1, nx].

        Returns
        -------
        tuple[MomentumCoefficients, MomentumCoefficients]
            The u equation in shape [ny, nx+1] and the v equation in shape
            [ny+1, nx].
        """
        self._check_shapes(u, v)
        c_u = self._assemble(u, v, self._for_u)
        c_v = _transpose(self._assemble(u=v.T, v=u.T, o=self._for_v))
        return c_u, c_v

    def predict(
        self,
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        mu_eff: np.ndarray | None = None,
        wall_mu: dict[str, np.ndarray] | None = None,
    ) -> MomentumPrediction:
        """Under-relaxed Jacobi sweeps of both momentum equations.

        Parameters
        ----------
        u : np.ndarray
            x-velocity, shape [ny, nx+1], boundary columns already set.
        v : np.ndarray
            y-velocity, shape [ny+1, nx], boundary rows already set.
        p : np.ndarray
            Pressure at cell centers, shape [ny, nx].
        mu_eff : np.ndarray, optional
            Dynamic effective viscosity per cell, Pa s, shape [ny, nx],
            float64, finite and positive in every non-SOLID cell; SOLID
            cells are not read. None runs the scalar path with air's
            viscosity, bitwise the path without the argument.
        wall_mu : dict[str, np.ndarray], optional
            The viscosity of the wall stencil, per wall face, in place of
            air's or the field's. Keys "u" and "v", each float64 of shape
            [ny+1, nx+1] indexed by corner: entry ``[j, i]`` is the face
            through ``(x[i], y[j])`` that the component's wall stencil
            crosses, horizontal for "u" (bottom and top edges) and vertical
            for "v" (left and right edges). Read only at the faces whose
            tangential condition is a Dirichlet value and that border an
            unknown; there it must be finite and positive, and elsewhere it
            is not read. None uses air's viscosity or the field's.

        Returns
        -------
        MomentumPrediction
            u*, v* and the un-relaxed diagonals; see that class.

        Raises
        ------
        TypeError
            If ``mu_eff`` or a ``wall_mu`` entry is not a float64 ndarray, or
            ``wall_mu`` is not a dict.
        ValueError
            On a wrong shape, a non-finite or non-positive value where one is
            read, or ``wall_mu`` keys other than "u" and "v".

        Notes
        -----
        Under-relaxation follows the collocated solver: the diagonal is
        divided by alpha_velocity and ``(1 - alpha) / alpha * a_P * phi`` is
        added to the source, so the returned ``a_P`` is the un-relaxed one.
        The pressure source is ``-(p_(s+) - p_(s-)) * face_width``, the
        pressure difference across the control volume, which on the
        staggered grid needs no interpolation. ``solver.momentum_sweeps``
        sweeps run on one assembly; see ``_sweep_n``.
        """
        self._check_shapes(u, v)
        if p.shape != (self._u_shape[0], self._v_shape[1]):
            raise ValueError(
                f"expected p of shape {(self._u_shape[0], self._v_shape[1])}, got {p.shape}"
            )
        mu_cells = None if mu_eff is None else self._check_mu_eff(mu_eff)
        walls = None if wall_mu is None else self._check_wall_mu(wall_mu)
        c_u = self._assemble(
            u,
            v,
            self._for_u,
            mu_cells=mu_cells,
            wall=None if walls is None else walls["u"],
        )
        u_star = self._sweep_n(u, c_u, self._pressure_source(p, self._for_u))

        c_vt = self._assemble(
            u=v.T,
            v=u.T,
            o=self._for_v,
            mu_cells=None if mu_cells is None else np.ascontiguousarray(mu_cells.T),
            wall=None if walls is None else np.ascontiguousarray(walls["v"].T),
        )
        v_star = self._sweep_n(v.T, c_vt, self._pressure_source(p.T, self._for_v)).T
        return MomentumPrediction(
            u_star=np.ascontiguousarray(u_star),
            v_star=np.ascontiguousarray(v_star),
            a_p_u=c_u.a_p,
            a_p_v=np.ascontiguousarray(c_vt.a_p.T),
        )

    # ------------------------------------------------------------------
    # Assembly in the component's own frame: transverse rows, own axis last
    # ------------------------------------------------------------------

    def _check_shapes(self, u: np.ndarray, v: np.ndarray) -> None:
        if u.shape != self._u_shape or v.shape != self._v_shape:
            raise ValueError(
                f"expected staggered shapes u {self._u_shape} and v {self._v_shape}, "
                f"got u {u.shape} and v {v.shape}"
            )

    def _check_mu_eff(self, mu_eff: np.ndarray) -> np.ndarray:
        """The viscosity field as the assembly reads it, or a refusal before any arithmetic.

        Returns a copy with SOLID cells set to air's viscosity: the face rule
        never reads them at an unknown, and a positive value there keeps the
        harmonic mean defined on faces that are not unknowns.
        """
        if not isinstance(mu_eff, np.ndarray) or mu_eff.dtype != np.float64:
            got = (
                mu_eff.dtype
                if isinstance(mu_eff, np.ndarray)
                else type(mu_eff).__name__
            )
            raise TypeError(f"mu_eff must be a float64 ndarray, got {got}")
        shape = (self._u_shape[0], self._v_shape[1])
        if mu_eff.shape != shape:
            raise ValueError(f"expected mu_eff of shape {shape}, got {mu_eff.shape}")
        live = mu_eff[self._live]
        if not np.isfinite(live).all():
            raise ValueError("mu_eff must be finite in every non-SOLID cell")
        if np.any(live <= 0.0):
            raise ValueError("mu_eff must be positive in every non-SOLID cell")
        return np.where(self._live, mu_eff, self._mu)

    def _check_wall_mu(self, wall_mu: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The wall viscosities as given, or a refusal before any arithmetic."""
        if not isinstance(wall_mu, dict):
            raise TypeError(
                f"wall_mu must be a dict with keys 'u' and 'v', got {type(wall_mu).__name__}"
            )
        if set(wall_mu) != {"u", "v"}:
            raise ValueError(
                f"wall_mu must have keys 'u' and 'v', got {sorted(wall_mu)}"
            )
        shape = (self._u_shape[0] + 1, self._v_shape[1] + 1)
        for key in ("u", "v"):
            field = wall_mu[key]
            if not isinstance(field, np.ndarray) or field.dtype != np.float64:
                got = (
                    field.dtype
                    if isinstance(field, np.ndarray)
                    else type(field).__name__
                )
                raise TypeError(
                    f"wall_mu['{key}'] must be a float64 ndarray, got {got}"
                )
            if field.shape != shape:
                raise ValueError(
                    f"expected wall_mu['{key}'] of shape {shape}, got {field.shape}"
                )
            read = field[self._wall_read[key]]
            if not np.isfinite(read).all():
                raise ValueError(f"wall_mu['{key}'] must be finite at every wall face")
            if np.any(read <= 0.0):
                raise ValueError(
                    f"wall_mu['{key}'] must be positive at every wall face"
                )
        return wall_mu

    @staticmethod
    def _wall_faces(o: _Orientation) -> np.ndarray:
        """Transverse faces whose wall viscosity an unknown reads, [nt+1, ns+1], own frame."""
        nt, ns = o.solid.shape
        unknown = ~(o.solid[:, :-1] | o.solid[:, 1:])
        read = np.zeros((nt + 1, ns + 1), dtype=bool)
        read[0, 1:-1] = o.low.is_dirichlet[1:-1] & unknown[0, :]
        read[-1, 1:-1] = o.high.is_dirichlet[1:-1] & unknown[-1, :]
        return read

    def _assemble(
        self,
        u: np.ndarray,
        v: np.ndarray,
        o: _Orientation,
        mu_cells: np.ndarray | None = None,
        wall: np.ndarray | None = None,
    ) -> MomentumCoefficients:
        """Coefficients of the component ``u`` whose own axis is the last one.

        ``u`` has shape [nt, ns+1] and ``v``, the other component, shape
        [nt+1, ns]. For the v equation the caller passes both transposed,
        with ``mu_cells`` [nt, ns] and ``wall`` [nt+1, ns+1] in the same
        frame. With ``mu_cells`` None the scalar lines run.
        """
        nt, ns1 = u.shape
        ns = ns1 - 1
        rho, mu = self._rho, self._mu
        block = (slice(None), slice(1, ns))

        # Mass fluxes: streamwise faces at s_centers [nt, ns]; transverse
        # faces at t_faces [nt+1, ns+1], columns 0 and ns unused.
        f_s = rho * 0.5 * (u[:, :-1] + u[:, 1:]) * o.dt_cell[:, None]
        f_t = np.zeros((nt + 1, ns + 1), dtype=np.float64)
        f_t[:, 1:-1] = (
            rho
            * 0.5
            * (v[:, :-1] * o.ds_cell[None, :-1] + v[:, 1:] * o.ds_cell[None, 1:])
        )

        # Diffusion conductances. The two transverse edge rows divide by the
        # wall distance the boundary layer reports and carry nothing across
        # a zero-gradient edge.
        d_t = np.empty((nt + 1, ns + 1), dtype=np.float64)
        if mu_cells is None:
            d_s = mu * o.dt_cell[:, None] / o.ds_cell[None, :]
            d_t[1:-1, :] = mu * o.ds_face[None, :] / o.dt_face[1:-1, None]
            mu_low = mu_high = mu
        else:
            corner = self._corner_viscosity(mu_cells, o)
            d_s = mu_cells * o.dt_cell[:, None] / o.ds_cell[None, :]
            d_t[1:-1, :] = corner[1:-1, :] * o.ds_face[None, :] / o.dt_face[1:-1, None]
            mu_low, mu_high = corner[0, :], corner[-1, :]
        if wall is not None:
            mu_low, mu_high = wall[0, :], wall[-1, :]
        d_t[0, :] = np.where(
            o.low.is_dirichlet, mu_low * o.ds_face / o.low.wall_distance, 0
        )
        d_t[-1, :] = np.where(
            o.high.is_dirichlet, mu_high * o.ds_face / o.high.wall_distance, 0
        )

        # Upwind coefficients on the unknown block [nt, ns-1]
        f_e, f_w = f_s[:, 1:], f_s[:, :-1]
        f_n, f_sth = f_t[1:, 1:-1], f_t[:-1, 1:-1]
        a_e = d_s[:, 1:] + np.maximum(-f_e, 0.0)
        a_w = d_s[:, :-1] + np.maximum(f_w, 0.0)
        a_n = d_t[1:, 1:-1] + np.maximum(-f_n, 0.0)
        a_sth = d_t[:-1, 1:-1] + np.maximum(f_sth, 0.0)
        a_p = a_e + a_w + a_n + a_sth + (f_e - f_w + f_n - f_sth)

        # Transverse edges: a Dirichlet value is a known neighbour and moves
        # to the source; a zero-gradient edge has phi_P as its neighbour, so
        # that coefficient leaves the diagonal. Either way no stored
        # neighbour exists beyond the edge.
        hi = o.high.is_dirichlet[1:-1]
        lo = o.low.is_dirichlet[1:-1]
        b_boundary = np.zeros((nt, ns - 1), dtype=np.float64)
        b_boundary[-1, :] += np.where(hi, a_n[-1, :] * o.high.value[1:-1], 0.0)
        b_boundary[0, :] += np.where(lo, a_sth[0, :] * o.low.value[1:-1], 0.0)
        a_p[-1, :] -= np.where(hi, 0.0, a_n[-1, :])
        a_p[0, :] -= np.where(lo, 0.0, a_sth[0, :])
        a_n[-1, :] = 0.0
        a_sth[0, :] = 0.0

        b_deferred = self._deferred_correction(u, f_s, f_t, o)
        if mu_cells is not None:
            b_deferred = b_deferred + self._stress_source(u, v, o, mu_cells, corner)

        # Faces of SOLID cells are not unknowns
        solid_face = o.solid[:, :-1] | o.solid[:, 1:]
        for arr in (a_p, a_e, a_w, a_n, a_sth, b_boundary, b_deferred):
            arr[solid_face] = 0.0

        def full(arr: np.ndarray) -> np.ndarray:
            out = np.zeros((nt, ns + 1), dtype=np.float64)
            out[block] = arr
            return out

        return MomentumCoefficients(
            a_p=full(a_p),
            a_s_plus=full(a_e),
            a_s_minus=full(a_w),
            a_t_plus=full(a_n),
            a_t_minus=full(a_sth),
            b_boundary=full(b_boundary),
            b_deferred=full(b_deferred),
        )

    def _deferred_correction(
        self, u: np.ndarray, f_s: np.ndarray, f_t: np.ndarray, o: _Orientation
    ) -> np.ndarray:
        """Minus the net (QUICK - upwind) advective flux, on the unknown block."""
        nt, ns1 = u.shape
        ns = ns1 - 1

        # Streamwise faces at s_centers, nodes at s_faces, boundary columns
        # included as nodes.
        pos_s = f_s > 0.0
        q_s = quick_face_values(u, o.s_faces, np.arange(ns), o.s_centers, pos_s)
        up_s = np.where(pos_s, u[:, :-1], u[:, 1:])
        dq_s = (q_s - up_s) * f_s

        # Interior transverse faces at t_faces[1..nt-1]; the nodes are the
        # rows of u plus one node on each edge holding the wall value or,
        # across a zero-gradient edge, the adjacent value. The edge faces
        # themselves take the edge value under both schemes and contribute
        # nothing.
        low = np.where(o.low.is_dirichlet, o.low.value, u[0, :])
        high = np.where(o.high.is_dirichlet, o.high.value, u[-1, :])
        ext = np.vstack([low, u, high])
        t_nodes = np.concatenate(([o.t_faces[0]], o.t_centers, [o.t_faces[-1]]))
        pos_t = f_t[1:-1, :] > 0.0
        q_t = quick_face_values(
            ext.T, t_nodes, np.arange(1, nt), o.t_faces[1:-1], pos_t.T
        ).T
        up_t = np.where(pos_t, u[:-1, :], u[1:, :])
        dq_t = np.zeros((nt + 1, ns + 1), dtype=np.float64)
        dq_t[1:-1, :] = (q_t - up_t) * f_t[1:-1, :]

        return -(dq_s[:, 1:] - dq_s[:, :-1] + dq_t[1:, 1:-1] - dq_t[:-1, 1:-1])

    # ------------------------------------------------------------------
    # The viscosity field (ADR-012 D), in the component's own frame: cells
    # [nt, ns]. The arithmetic is frozen34.py's, line for line, so the two
    # agree to the bit.
    # ------------------------------------------------------------------

    @staticmethod
    def _corner_viscosity(mu: np.ndarray, o: _Orientation) -> np.ndarray:
        """mu at every transverse face of the component, shape [nt+1, ns+1].

        Harmonic across the row boundary in each column, distance-weighted;
        then arithmetic across the two columns, width-weighted. A SOLID cell
        takes the value of the non-SOLID cell across the row boundary.
        """
        nt, ns = mu.shape
        solid = o.solid
        lo, hi = mu[:-1, :], mu[1:, :]
        lo_eff = np.where(solid[:-1, :] & ~solid[1:, :], hi, lo)
        hi_eff = np.where(solid[1:, :] & ~solid[:-1, :], lo, hi)
        d_lo = o.t_faces[1:-1] - o.t_centers[:-1]
        d_hi = o.t_centers[1:] - o.t_faces[1:-1]
        frac_lo = (d_lo / (d_lo + d_hi))[:, None]
        rows = np.empty((nt + 1, ns), dtype=np.float64)
        rows[1:-1, :] = _harmonic(lo_eff, hi_eff, frac_lo)
        rows[0, :] = mu[0, :]
        rows[-1, :] = mu[-1, :]
        w_l = o.s_faces[1:-1] - o.s_centers[:-1]
        w_r = o.s_centers[1:] - o.s_faces[1:-1]
        frac_r = w_r / (w_l + w_r)
        corner = np.empty((nt + 1, ns + 1), dtype=np.float64)
        corner[:, 1:-1] = _lerp(rows[:, :-1], rows[:, 1:], frac_r[None, :])
        corner[:, 0] = rows[:, 0]
        corner[:, -1] = rows[:, -1]
        return corner

    @staticmethod
    def _face_viscosity(mu: np.ndarray, o: _Orientation) -> np.ndarray:
        """mu at the component's own interior faces, [nt, ns-1]: the two cells' width mean."""
        w_l = o.s_faces[1:-1] - o.s_centers[:-1]
        w_r = o.s_centers[1:] - o.s_faces[1:-1]
        return _lerp(mu[:, :-1], mu[:, 1:], (w_r / (w_l + w_r))[None, :])

    def _stress_source(
        self,
        u: np.ndarray,
        v: np.ndarray,
        o: _Orientation,
        mu: np.ndarray,
        corner: np.ndarray,
    ) -> np.ndarray:
        """Form b of the stress terms a varying viscosity adds, on the unknown block [nt, ns-1].

        du/dx is differenced at cell centres and dv/dx at the corners, the
        two domain-edge rows included; there the corner value equals the
        face's own, so form b has no edge term.
        """
        ds_f = o.ds_face[1:-1]
        grad_s = (u[:, 1:] - u[:, :-1]) / o.ds_cell[None, :]
        grad_t = (v[:, 1:] - v[:, :-1]) / ds_f[None, :]
        flux_s = grad_s * o.dt_cell[:, None]
        flux_t = grad_t * ds_f[None, :]
        mu_f = self._face_viscosity(mu, o)
        mu_c = corner[:, 1:-1]
        return (
            (mu[:, 1:] - mu_f) * flux_s[:, 1:]
            - (mu[:, :-1] - mu_f) * flux_s[:, :-1]
            + (mu_c[1:, :] - mu_f) * flux_t[1:, :]
            - (mu_c[:-1, :] - mu_f) * flux_t[:-1, :]
        )

    @staticmethod
    def _pressure_source(p: np.ndarray, o: _Orientation) -> np.ndarray:
        """-(p_(s+) - p_(s-)) times the transverse face width, full component shape."""
        nt, ns = p.shape
        b = np.zeros((nt, ns + 1), dtype=np.float64)
        b[:, 1:-1] = -(p[:, 1:] - p[:, :-1]) * o.dt_cell[:, None]
        return b

    def _sweep(
        self, phi: np.ndarray, c: MomentumCoefficients, b_pressure: np.ndarray
    ) -> np.ndarray:
        """One under-relaxed Jacobi sweep; non-unknown faces keep their value or zero."""
        alpha = self._alpha
        unknown = c.a_p > 0.0
        a_p_ur = np.where(unknown, c.a_p / alpha, 1.0)
        b = (
            c.b_boundary
            + c.b_deferred
            + b_pressure
            + (1.0 - alpha) / alpha * c.a_p * phi
        )

        padded = np.pad(phi, ((1, 1), (1, 1)))
        numerator = (
            c.a_s_plus * padded[1:-1, 2:]
            + c.a_s_minus * padded[1:-1, :-2]
            + c.a_t_plus * padded[2:, 1:-1]
            + c.a_t_minus * padded[:-2, 1:-1]
            + b
        )
        phi_new = phi.copy()
        interior = np.zeros_like(unknown)
        interior[:, 1:-1] = True
        phi_new[interior] = np.where(unknown, numerator / a_p_ur, 0.0)[interior]
        return phi_new

    def _sweep_n(
        self, phi: np.ndarray, c: MomentumCoefficients, b_pressure: np.ndarray
    ) -> np.ndarray:
        """``momentum_sweeps`` Jacobi sweeps of the under-relaxed equations.

        One sweep is ``_sweep``. More hold the coefficients, the sources and
        the relaxation term ``(1 - alpha) / alpha * a_P * phi`` at the outer
        iterate ``phi`` and iterate only the neighbour values, as frozen34.py
        did, so the sweeps converge toward the solution of the under-relaxed
        linear system of this outer iteration.
        """
        if self._n_sweeps == 1:
            return self._sweep(phi, c, b_pressure)
        alpha = self._alpha
        unknown = c.a_p > 0.0
        a_p_ur = np.where(unknown, c.a_p / alpha, 1.0)
        b = (
            c.b_boundary
            + c.b_deferred
            + b_pressure
            + (1.0 - alpha) / alpha * c.a_p * phi
        )
        interior = np.zeros_like(unknown)
        interior[:, 1:-1] = True
        current = phi
        for _ in range(self._n_sweeps):
            padded = np.pad(current, ((1, 1), (1, 1)))
            numerator = (
                c.a_s_plus * padded[1:-1, 2:]
                + c.a_s_minus * padded[1:-1, :-2]
                + c.a_t_plus * padded[2:, 1:-1]
                + c.a_t_minus * padded[:-2, 1:-1]
                + b
            )
            following = phi.copy()
            following[interior] = np.where(unknown, numerator / a_p_ur, 0.0)[interior]
            current = following
        return current


def _transpose(c: MomentumCoefficients) -> MomentumCoefficients:
    """Return the coefficients with every array transposed and contiguous."""
    t = np.ascontiguousarray
    return MomentumCoefficients(
        a_p=t(c.a_p.T),
        a_s_plus=t(c.a_s_plus.T),
        a_s_minus=t(c.a_s_minus.T),
        a_t_plus=t(c.a_t_plus.T),
        a_t_minus=t(c.a_t_minus.T),
        b_boundary=t(c.b_boundary.T),
        b_deferred=t(c.b_deferred.T),
    )
