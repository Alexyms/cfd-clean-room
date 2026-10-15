"""Additive-correction multigrid: a preconditioner for the p' equation's conjugate gradients.

ADR-013 decision 1 names multigrid preconditioning of the same CG loop as the
successor to the diagonal preconditioner; this is that successor (prompt 53,
2026-10-14). It is aggregation multigrid in the form Hutchinson and Raithby
(1986) gave for SIMPLE pressure corrections, applied as one V-cycle per CG
iteration. The module knows the five-point system and nothing else of the
solver: ``Multigrid`` is built from a coefficient set and the operator that
applies it, and ``cycle`` maps a residual to a preconditioned vector.

The hierarchy. Each coarser level groups the cells of the one below in 2x2
blocks; on a side with an odd count the last block is one cell wide.
Prolongation P is piecewise constant (each fine cell takes its group's value)
and restriction is its transpose, a sum over the group. With that P the
Galerkin operator ``P^T A P`` is again a five-point stencil and needs no
probing: a coarse face coefficient is the sum of the fine face coefficients
crossing that coarse face, and a coarse diagonal is the sum of its children's
diagonals minus twice the couplings inside the group. SOLID cells (zero rows),
obstacles, the outlet terms in the diagonal and a closed domain's constant
null space all reach every level through those sums. A coarse cell has an
equation when any child has one. Coarsening stops at COARSEST_CELLS cells or
when a side reaches one cell, and the coarsest level is solved exactly by a
dense pseudo-inverse, which on a closed domain returns the minimum-norm
solution.

The cycle. A V(nu, nu) cycle from zero with weighted Jacobi before and after
the coarse correction, the same weight and sweep count both ways, so the
cycle is a symmetric operator, and positive definite on the range while the
weighted sweep is a contraction in the energy norm (``omega lambda_max <
2``), as conjugate gradients requires. Every operation is a whole-array NumPy
expression: the per-cell structure of decision 1's GPU path holds on every
level, and the coarsest solve is one small dense product.

The weight. Every level's operator is weakly diagonally dominant: the finest
because a cell's diagonal is its couplings plus any outlet term, and each
coarser one because the Galerkin diagonal, the children's diagonals less
twice the couplings inside the group, leaves at least the couplings that
leave the group. By Gershgorin the eigenvalues of ``D^-1 A`` lie in [0, 2] on
every level, so the fixed weight ``omega = SMOOTHER_FACTOR / 2`` (2/3, the
classic damping for the five-point stencil) gives ``omega lambda_max <= 4/3``,
inside the limit, with no estimate of lambda_max. Prompt 53 built three power
iterations per level for that estimate and they returned 2 to within a
percent on every level of every system measured; prompt 54 replaced them.
``Multigrid`` refuses a finest system that is not weakly diagonally dominant,
since the bound would no longer hold.

The cost this design removes. The geometric multigrid of ADR-013's probe
(docs/reports/pressure_solver_ecr003.md, section 7.3) formed its coarse
operators by probing with 25 coloured vectors per level, 120 to 175 ms per
correction on the product mesh, every correction, since the coefficients
change every outer iteration. Here the setup is a few strided sums per level.
"""

import dataclasses
from collections.abc import Callable
from typing import Protocol

import numpy as np

# Coarsening stops once a level has at most this many cells; that level is
# solved by a dense pseudo-inverse, whose cost (cubic in the cells) is small at
# this size against one smoothing sweep of the finest level. The value is the
# one ADR-013's probe used for its geometric multigrid (the report, section
# 2.3).
COARSEST_CELLS = 64

# Weighted Jacobi sweeps before and after each coarse correction: nu of the
# V(nu, nu) cycle. Chosen on two systems outside prompt 53's measurement set,
# the product room on 80x30 and the backward-facing step on 410x40
# (docs/reports/pressure_multigrid.md, section 2): V(2, 2) and V(3, 3) took
# one to seven fewer CG iterations than V(1, 1) but more time per correction,
# since each sweep is a full product on every level.
SMOOTHING_SWEEPS = 1

# The weight is SMOOTHER_FACTOR / 2: lambda_max(D^-1 A) is at most 2 on every
# level (module docstring), so omega lambda_max is at most SMOOTHER_FACTOR,
# inside the contraction limit 2.
SMOOTHER_FACTOR = 4.0 / 3.0

# Rounding allowed when testing a row for diagonal dominance: a diagonal is
# the sum of its couplings formed in some order, so it can fall short of them
# by a few units in the last place.
DOMINANCE_TOLERANCE = 1.0e-12


class FivePointCoefficients(Protocol):
    """The five arrays of a p' system, shape [ny, nx] each (``PressureCoefficients``)."""

    a_p: np.ndarray
    a_e: np.ndarray
    a_w: np.ndarray
    a_n: np.ndarray
    a_s: np.ndarray


@dataclasses.dataclass(frozen=True)
class Level:
    """One level of the hierarchy.

    Parameters
    ----------
    coefficients : FivePointCoefficients
        The level's operator, the Galerkin product of the one above (the
        given system on the finest level).
    active : np.ndarray
        Bool, [ny, nx]: the cells with an equation.
    smoother : np.ndarray
        ``omega / a_P`` where a_P > 0, zero elsewhere: one weighted Jacobi
        sweep is ``x += smoother * (r - A x)``.
    weight : float
        omega, ``SMOOTHER_FACTOR / 2`` on a smoothed level; zero on the
        coarsest, which is solved exactly.
    """

    coefficients: FivePointCoefficients
    active: np.ndarray
    smoother: np.ndarray
    weight: float


class Multigrid:
    """An additive-correction multigrid hierarchy and its V-cycle.

    Parameters
    ----------
    coefficients : FivePointCoefficients
        The finest system, a frozen dataclass of five [ny, nx] arrays with
        ``A`` symmetric and positive semi-definite, the edge coefficients
        (``a_w[:, 0]``, ``a_e[:, -1]``, ``a_s[0, :]``, ``a_n[-1, :]``) zero
        and every coefficient of a cell with no equation zero. Coarse levels
        are built with ``dataclasses.replace`` on it, so they have its type.
    apply : Callable[[FivePointCoefficients, np.ndarray], np.ndarray]
        The operator: ``apply(c, x)`` returns ``A x`` for coefficients c,
        ``pressure.apply_operator``. Every level uses it.

    Attributes
    ----------
    levels : list[Level]
        Finest first. The last is the coarsest, solved exactly.
    coarsest_inverse : np.ndarray
        The pseudo-inverse of the coarsest operator, [n, n] for its n cells
        in row-major order.

    Raises
    ------
    ValueError
        If the five arrays are not all two-dimensional and of one shape, or
        a row of the finest system is not weakly diagonally dominant (its
        diagonal below the sum of its four couplings), which the weight's
        bound needs.
    """

    def __init__(
        self,
        coefficients: FivePointCoefficients,
        apply: Callable[[FivePointCoefficients, np.ndarray], np.ndarray],
    ) -> None:
        shape = coefficients.a_p.shape
        arrays = (
            coefficients.a_p,
            coefficients.a_e,
            coefficients.a_w,
            coefficients.a_n,
            coefficients.a_s,
        )
        if len(shape) != 2 or any(a.shape != shape for a in arrays):
            raise ValueError(
                "expected five two-dimensional coefficient arrays of one shape, got "
                f"{[a.shape for a in arrays]}"
            )
        # The coarse levels inherit dominance from the finest through the
        # sums, so the finest is the only level that needs the check.
        c = coefficients
        couplings = (c.a_e + c.a_w) + (c.a_n + c.a_s)
        if np.any(c.a_p * (1.0 + DOMINANCE_TOLERANCE) < couplings):
            raise ValueError(
                "the finest system is not weakly diagonally dominant: a diagonal "
                "is below the sum of its couplings"
            )
        self._apply = apply
        self.levels: list[Level] = []
        active = coefficients.a_p > 0.0
        while True:
            ny, nx = c.a_p.shape
            coarsest = ny * nx <= COARSEST_CELLS or min(ny, nx) == 1
            self.levels.append(self._level(c, active, smoothed=not coarsest))
            if coarsest:
                break
            c, active = coarsen(c, active)
        self.coarsest_inverse: np.ndarray = _dense_pseudo_inverse(
            self.levels[-1].coefficients
        )

    def _level(
        self, c: FivePointCoefficients, active: np.ndarray, smoothed: bool
    ) -> Level:
        """One level, with the fixed weight on every level that is smoothed.

        The coarsest level is solved exactly and never smoothed, so its
        weight is zero.
        """
        has_diagonal = c.a_p > 0.0
        inverse = np.where(has_diagonal, 1.0 / np.where(has_diagonal, c.a_p, 1.0), 0.0)
        weight = SMOOTHER_FACTOR / 2.0 if smoothed else 0.0
        return Level(
            coefficients=c, active=active, smoother=weight * inverse, weight=weight
        )

    def cycle(self, r: np.ndarray) -> np.ndarray:
        """One V(nu, nu) cycle from zero on ``A z = r``.

        Parameters
        ----------
        r : np.ndarray
            A residual, the finest level's shape, zero at cells with no
            equation; not modified.

        Returns
        -------
        np.ndarray
            z, the finest level's shape, zero at cells with no equation. The
            map ``r -> z`` is linear, symmetric and positive semi-definite.

        Notes
        -----
        A cell with no equation has a zero row and column on every level and
        a zero smoother, so whatever value the prolongation gives it is never
        read by a cell that has one; it is zeroed once, here, rather than on
        every level.
        """
        z = self._cycle(0, r)
        z[~self.levels[0].active] = 0.0
        return z

    def _cycle(self, index: int, r: np.ndarray) -> np.ndarray:
        """The V-cycle on level ``index`` and below."""
        if index == len(self.levels) - 1:
            return (self.coarsest_inverse @ r.ravel()).reshape(r.shape)
        level = self.levels[index]
        c = level.coefficients
        x = level.smoother * r
        for _ in range(SMOOTHING_SWEEPS - 1):
            x += level.smoother * (r - self._apply(c, x))
        coarse = self._cycle(index + 1, restrict(r - self._apply(c, x)))
        add_prolonged(x, coarse)
        for _ in range(SMOOTHING_SWEEPS):
            x += level.smoother * (r - self._apply(c, x))
        return x


def coarsen(
    c: FivePointCoefficients, active: np.ndarray
) -> tuple[FivePointCoefficients, np.ndarray]:
    """The Galerkin operator ``P^T A P`` of piecewise-constant 2x2 aggregation.

    Parameters
    ----------
    c : FivePointCoefficients
        The fine system, a frozen dataclass (see ``Multigrid``).
    active : np.ndarray
        Bool, the fine cells with an equation.

    Returns
    -------
    tuple[FivePointCoefficients, np.ndarray]
        The coarse system, of c's type, shape [ceil(ny/2), ceil(nx/2)], and
        its cells with an equation: those with at least one fine child that
        has one.

    Notes
    -----
    A coarse face coefficient sums the fine coefficients across that face:
    the east one, ``a_e`` of the group's right column over its two rows. The
    couplings inside a group are ``a_e`` of its left column and ``a_n`` of its
    bottom row; each appears in two children's diagonals and in two
    off-diagonal entries of the group's block, so the coarse diagonal is the
    children's diagonals minus twice their sum. An odd side is padded with a
    zero row or column, a cell with no coefficients, so a lone cell's group
    is the cell.
    """
    a_p, a_e, a_w, a_n, a_s = (_pad(a) for a in (c.a_p, c.a_e, c.a_w, c.a_n, c.a_s))
    inside = _rows(a_e[:, 0::2]) + _columns(a_n[0::2, :])
    coarse = dataclasses.replace(
        c,
        a_p=_rows(_columns(a_p)) - 2.0 * inside,
        a_e=_rows(a_e[:, 1::2]),
        a_w=_rows(a_w[:, 0::2]),
        a_n=_columns(a_n[1::2, :]),
        a_s=_columns(a_s[0::2, :]),
    )
    return coarse, restrict(active.astype(np.float64)) > 0.0


def restrict(r: np.ndarray) -> np.ndarray:
    """``P^T r``: the sum over each 2x2 group.

    Parameters
    ----------
    r : np.ndarray
        A fine-level vector, shape [ny, nx]; not modified.

    Returns
    -------
    np.ndarray
        The group sums, shape [ceil(ny/2), ceil(nx/2)].

    Notes
    -----
    The four children of a group are the four strided quarters of r; on an
    odd side the last group has no second child there, so the second
    quarters are one row or column short and add into the leading part.
    """
    ny, nx = r.shape
    out = r[0::2, 0::2].copy()
    out[: ny // 2, :] += r[1::2, 0::2]
    out[:, : nx // 2] += r[0::2, 1::2]
    out[: ny // 2, : nx // 2] += r[1::2, 1::2]
    return out


def add_prolonged(x: np.ndarray, z: np.ndarray) -> None:
    """``x += P z`` in place: each fine cell of x gains its group's value in z.

    Parameters
    ----------
    x : np.ndarray
        A fine-level vector, shape [ny, nx]; modified in place.
    z : np.ndarray
        The coarse-level vector, shape [ceil(ny/2), ceil(nx/2)]; not
        modified.

    Returns
    -------
    None
        The result is in x.
    """
    ny, nx = x.shape
    x[0::2, 0::2] += z
    x[1::2, 0::2] += z[: ny // 2, :]
    x[0::2, 1::2] += z[:, : nx // 2]
    x[1::2, 1::2] += z[: ny // 2, : nx // 2]


def _pad(a: np.ndarray) -> np.ndarray:
    """a with a zero row and column appended where its side is odd."""
    ny, nx = a.shape
    if ny % 2 == 0 and nx % 2 == 0:
        return a
    padded = np.zeros((ny + ny % 2, nx + nx % 2), dtype=a.dtype)
    padded[:ny, :nx] = a
    return padded


def _rows(a: np.ndarray) -> np.ndarray:
    """Sum of each pair of rows, (2m, n) -> (m, n)."""
    return a[0::2, :] + a[1::2, :]


def _columns(a: np.ndarray) -> np.ndarray:
    """Sum of each pair of columns, (m, 2n) -> (m, n)."""
    return a[:, 0::2] + a[:, 1::2]


def _dense_pseudo_inverse(c: FivePointCoefficients) -> np.ndarray:
    """The pseudo-inverse of the five-point operator as a dense matrix, row-major cells.

    The matrix is symmetric, so the symmetric eigensolver forms it. On a
    closed domain the constant over the cells with an equation is its null
    space and the pseudo-inverse returns the minimum-norm solution; rows and
    columns of cells with no equation are zero and stay zero.
    """
    ny, nx = c.a_p.shape
    n = ny * nx
    index = np.arange(n).reshape(ny, nx)
    matrix = np.zeros((n, n))
    matrix[index.ravel(), index.ravel()] = c.a_p.ravel()
    for coefficient, own, other in (
        (c.a_e[:, :-1], index[:, :-1], index[:, 1:]),
        (c.a_w[:, 1:], index[:, 1:], index[:, :-1]),
        (c.a_n[:-1, :], index[:-1, :], index[1:, :]),
        (c.a_s[1:, :], index[1:, :], index[:-1, :]),
    ):
        matrix[own.ravel(), other.ravel()] -= coefficient.ravel()
    return np.linalg.pinv(matrix, hermitian=True)
