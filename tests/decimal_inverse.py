"""The pseudo-inverse of a symmetric matrix in 60-digit decimals: the referee of the coarsest solve.

The core's pseudo-inverse and NumPy's are each judged against this one, on the
scale of the system's condition (Alex, 2026-11-24; docs/reports/phase6_pressure.md,
section 4). Nothing here is the core's method or LAPACK's: the vectors NumPy's cutoff
drops are refined from LAPACK's by subspace inverse iteration, deflated, and what is
left is inverted by LU factors.
"""

import dataclasses
import decimal
from collections.abc import Iterable

import numpy as np

Matrix = list[list[decimal.Decimal]]

DIGITS = 60
# The shift that keeps the factors of a singular matrix finite, over its largest
# diagonal: far below what the cutoff drops and far above the rounding.
SHIFT = decimal.Decimal("1e-25")
# The iteration has converged when a step moves the subspace by less than this.
CONVERGED = decimal.Decimal("1e-45")
MAX_STEPS = 60


class RefereeError(ArithmeticError):
    """A matrix the referee does not judge, with its reason."""


@dataclasses.dataclass(frozen=True, eq=False)
class Referee:
    """A pseudo-inverse in decimals and what the criterion reads of it.

    Attributes
    ----------
    exact : list[list[Decimal]]
        The pseudo-inverse to the working precision, n by n.
    inverse : np.ndarray
        The same rounded to doubles, shape [n, n].
    condition : float
        The largest kept eigenvalue's magnitude over the smallest kept one's; 0 for
        a matrix with none.
    dropped : int
        The eigenvalues the cutoff drops.
    largest_dropped : float
        The largest dropped magnitude over the largest one; 0 with none dropped.
    """

    exact: Matrix
    inverse: np.ndarray
    condition: float
    dropped: int
    largest_dropped: float

    def bound(self, factor: float) -> float:
        """``factor n kappa eps max|X|``, the criterion's scale on this system."""
        eps = float(np.finfo(np.float64).eps)
        largest = float(np.abs(self.inverse).max())
        return factor * len(self.inverse) * self.condition * eps * largest


def referee(matrix: np.ndarray, rcond: float) -> Referee:
    """Form the pseudo-inverse of a symmetric matrix, by NumPy's cutoff, in decimals.

    Parameters
    ----------
    matrix : np.ndarray
        Shape [n, n]; the lower triangle is read, as ``np.linalg.eigh`` reads it.
    rcond : float
        The cutoff over the largest magnitude: an eigenvalue at or below it is
        dropped.

    Returns
    -------
    Referee
        The pseudo-inverse and its condition on the kept range.

    Raises
    ------
    RefereeError
        If the subspace does not converge, or the refined values do not fall
        either side of the cutoff as LAPACK's do.
    """
    n = len(matrix)
    values, vectors = np.linalg.eigh(matrix)
    magnitudes = np.abs(values)
    if not magnitudes.any():
        zero = np.zeros((n, n))
        return Referee([[decimal.Decimal(0)] * n for _ in range(n)], zero, 0.0, n, 0.0)
    null = magnitudes <= rcond * magnitudes.max()
    with decimal.localcontext() as context:
        context.prec = DIGITS
        a = [
            [decimal.Decimal(float(matrix[max(i, j), min(i, j)])) for j in range(n)]
            for i in range(n)
        ]
        scale = max(abs(a[i][i]) for i in range(n))
        dropped = _subspace(a, vectors[:, null].T, SHIFT * scale)
        images = [_apply(a, v) for v in dropped]
        block = [[_dot(v, w) for w in images] for v in dropped]
        deflated = _deflated(a, dropped, images, block)
        for v in dropped:
            for row, x in zip(deflated, v, strict=True):
                _add(row, scale * x, v)
        exact = _inverse(deflated)
        for v in dropped:
            for row, x in zip(exact, v, strict=True):
                _add(row, -x / scale, v)
        left = [abs(x) for x in np.linalg.eigvalsh(_rounded(block))] if dropped else []
    inverse = _rounded(exact)
    largest = float(magnitudes.max())
    smallest_kept = 1.0 / float(np.abs(np.linalg.eigvalsh(inverse)).max())
    if any(x > rcond * largest for x in left) or smallest_kept <= rcond * largest:
        raise RefereeError("the refined eigenvalues cross the cutoff LAPACK's do not")
    return Referee(
        exact,
        inverse,
        largest / smallest_kept,
        len(dropped),
        max(left, default=0.0) / largest,
    )


def residuals(
    matrix: np.ndarray, judged: Referee, rcond: float
) -> tuple[float, float, float]:
    """Moore and Penrose's conditions on a referee's inverse, each over its own scale.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix it was formed from.
    judged : Referee
        The referee.
    rcond : float
        The cutoff it was formed with.

    Returns
    -------
    tuple[float, float, float]
        ``max|X A X - X| / max|X|``, ``max|A X A - A| / max|A|`` with A the matrix
        on the kept range, and ``max|X - X^T| / max|X|``.
    """
    n = len(matrix)
    magnitudes = np.abs(np.linalg.eigvalsh(matrix))
    if not magnitudes.any():
        return 0.0, 0.0, 0.0
    vectors = np.linalg.eigh(matrix)[1]
    null = magnitudes <= rcond * magnitudes.max()
    with decimal.localcontext() as context:
        context.prec = DIGITS
        a = [
            [decimal.Decimal(float(matrix[max(i, j), min(i, j)])) for j in range(n)]
            for i in range(n)
        ]
        scale = max(abs(a[i][i]) for i in range(n))
        dropped = _subspace(a, vectors[:, null].T, SHIFT * scale)
        images = [_apply(a, v) for v in dropped]
        block = [[_dot(v, w) for w in images] for v in dropped]
        kept = _deflated(a, dropped, images, block)
        x = judged.exact
        product = _multiply(x, kept)
        xax = _multiply(product, x)
        axa = _multiply(kept, product)
        transposed = [list(column) for column in zip(*x, strict=True)]
        return (
            float(_apart(xax, x) / _largest(x)),
            float(_apart(axa, kept) / _largest(kept)),
            float(_apart(transposed, x) / _largest(x)),
        )


def _dot(a: Iterable[decimal.Decimal], b: Iterable[decimal.Decimal]) -> decimal.Decimal:
    return sum((x * y for x, y in zip(a, b, strict=True)), decimal.Decimal(0))


def _add(
    row: list[decimal.Decimal], factor: decimal.Decimal, v: list[decimal.Decimal]
) -> None:
    """``row += factor v`` in place."""
    row[:] = [x + factor * y for x, y in zip(row, v, strict=True)]


def _apply(a: Matrix, v: list[decimal.Decimal]) -> list[decimal.Decimal]:
    return [_dot(row, v) for row in a]


def _multiply(a: Matrix, b: Matrix) -> Matrix:
    columns = [list(column) for column in zip(*b, strict=True)]
    return [[_dot(row, column) for column in columns] for row in a]


def _largest(a: Matrix) -> decimal.Decimal:
    return max(abs(x) for row in a for x in row)


def _apart(a: Matrix, b: Matrix) -> decimal.Decimal:
    pairs = (zip(p, q, strict=True) for p, q in zip(a, b, strict=True))
    return max(abs(x - y) for pair in pairs for x, y in pair)


def _rounded(a: Matrix) -> np.ndarray:
    return np.array([[float(x) for x in row] for row in a])


def _deflated(
    a: Matrix,
    dropped: Matrix,
    images: Matrix,
    block: Matrix,
) -> Matrix:
    """``(I - V V^T) A (I - V V^T)`` with V the dropped vectors."""
    out = [list(row) for row in a]
    for k, (v, w) in enumerate(zip(dropped, images, strict=True)):
        for row, x, y in zip(out, v, w, strict=True):
            _add(row, -x, w)
            _add(row, -y, v)
        for other, entry in zip(dropped, block[k], strict=True):
            for row, x in zip(out, v, strict=True):
                _add(row, entry * x, other)
    return out


def _subspace(a: Matrix, start: np.ndarray, shift: decimal.Decimal) -> Matrix:
    """The invariant subspace beside ``start``'s rows, orthonormal rows, by inverse iteration."""
    if not len(start):
        return []
    shifted = [list(row) for row in a]
    for i, row in enumerate(shifted):
        row[i] += shift
    factors = _factors(shifted)
    v = _orthonormal([[decimal.Decimal(float(x)) for x in row] for row in start])
    for _ in range(MAX_STEPS):
        w = _orthonormal([_solve(factors, row) for row in v])
        moved = decimal.Decimal(0)
        for row in w:
            rest = list(row)
            for other in v:
                _add(rest, -_dot(row, other), other)
            moved = max(moved, *(abs(x) for x in rest))
        v = w
        if moved <= CONVERGED:
            return v
    raise RefereeError("the dropped subspace did not converge")


def _orthonormal(rows: Matrix) -> Matrix:
    """Gram and Schmidt's method, twice over."""
    for _ in range(2):
        out: Matrix = []
        for row in rows:
            rest = list(row)
            for other in out:
                _add(rest, -_dot(rest, other), other)
            norm = _dot(rest, rest).sqrt()
            out.append([x / norm for x in rest])
        rows = out
    return rows


def _factors(a: Matrix) -> tuple[Matrix, list[int]]:
    """LU factors with partial pivoting, and the rows' order."""
    n = len(a)
    lu = [list(row) for row in a]
    order = list(range(n))
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(lu[i][k]))
        lu[k], lu[pivot] = lu[pivot], lu[k]
        order[k], order[pivot] = order[pivot], order[k]
        top = lu[k]
        for i in range(k + 1, n):
            row = lu[i]
            if not row[k]:
                continue
            factor = row[k] / top[k]
            row[k:] = [factor] + [
                x - factor * y for x, y in zip(row[k + 1 :], top[k + 1 :], strict=True)
            ]
    return lu, order


def _solve(
    factors: tuple[Matrix, list[int]], b: list[decimal.Decimal]
) -> list[decimal.Decimal]:
    lu, order = factors
    n = len(lu)
    x: list[decimal.Decimal] = []
    for i in range(n):
        x.append(b[order[i]] - _dot(lu[i][:i], x))
    for i in reversed(range(n)):
        x[i] = (x[i] - _dot(lu[i][i + 1 :], x[i + 1 :])) / lu[i][i]
    return x


def _inverse(a: Matrix) -> Matrix:
    """The inverse of a symmetric matrix, its rows the solutions of unit vectors."""
    n = len(a)
    factors = _factors(a)
    zero, one = decimal.Decimal(0), decimal.Decimal(1)
    return [
        _solve(factors, [one if i == j else zero for i in range(n)]) for j in range(n)
    ]
