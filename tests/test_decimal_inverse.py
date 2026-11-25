"""The referee of the pseudo-inverses, on matrices whose pseudo-inverse is known."""

import numpy as np
import pytest

from tests.decimal_inverse import RefereeError, referee, residuals

pytestmark = pytest.mark.unit

RCOND = 1e-15
CLOSE = 1e-15


def _path(cells: int) -> np.ndarray:
    """A closed line of unit couplings: one null vector, the constant."""
    matrix = np.diag(np.r_[1.0, np.full(cells - 2, 2.0), 1.0])
    return matrix - np.diag(np.ones(cells - 1), 1) - np.diag(np.ones(cells - 1), -1)


def test_an_invertible_matrix_has_its_inverse() -> None:
    """Two by two, by hand, with its condition."""
    judged = referee(np.array([[2.0, -1.0], [-1.0, 2.0]]), RCOND)
    expected = np.array([[2.0, 1.0], [1.0, 2.0]]) / 3.0
    assert np.abs(judged.inverse - expected).max() <= CLOSE
    assert judged.dropped == 0 and judged.largest_dropped == 0.0
    assert judged.condition == pytest.approx(3.0, rel=1e-14)


def test_a_closed_line_drops_its_null_vector() -> None:
    """Three cells in a row: the minimum-norm solutions by hand, and a null vector dropped."""
    judged = referee(_path(3), RCOND)
    expected = np.array([[5.0, -1.0, -4.0], [-1.0, 2.0, -1.0], [-4.0, -1.0, 5.0]]) / 9.0
    assert np.abs(judged.inverse - expected).max() <= CLOSE
    assert judged.dropped == 1
    assert judged.condition == pytest.approx(3.0, rel=1e-14)


def test_moore_and_penroses_conditions_hold_to_forty_digits() -> None:
    """Two closed lines and two cells with no equation: four null vectors."""
    cells = 12
    matrix = np.zeros((cells, cells))
    matrix[:6, :6] = _path(6)
    matrix[6:10, 6:10] = 1e-3 * _path(4)
    judged = referee(matrix, RCOND)
    assert judged.dropped == 4
    assert max(residuals(matrix, judged, RCOND)) <= 1e-40
    assert not judged.inverse[10:].any()


def test_the_cutoff_is_against_the_largest_magnitude() -> None:
    """A value either side of the cutoff on the diagonal, kept and dropped, negative ones too."""
    for small, kept in ((3e-15, True), (-3e-15, True), (3e-16, False), (-3e-16, False)):
        judged = referee(np.diag([1.0, small, -0.5]), RCOND)
        assert judged.dropped == (not kept)
        assert judged.inverse[1, 1] == (pytest.approx(1.0 / small) if kept else 0.0)
        assert judged.inverse[2, 2] == pytest.approx(-2.0)


def test_the_lower_triangle_is_read_as_numpy_reads_it() -> None:
    """What is above the diagonal changes nothing."""
    matrix = _path(5) + np.diag([1.0, 0.0, 0.0, 0.0, 0.0])
    spoilt = matrix + np.triu(np.ones((5, 5)), 1)
    expected = referee(matrix, RCOND).inverse
    assert np.array_equal(referee(spoilt, RCOND).inverse, expected)


def test_the_bound_is_the_criterions() -> None:
    """The factor times n, the condition, eps and the largest magnitude."""
    judged = referee(np.array([[2.0, -1.0], [-1.0, 2.0]]), RCOND)
    assert judged.bound(4.0) == pytest.approx(4.0 * 2 * 3.0 * 2.0**-52 * 2.0 / 3.0)


def test_a_zero_matrix_is_its_own() -> None:
    """Nothing is kept, so the condition is zero."""
    judged = referee(np.zeros((3, 3)), RCOND)
    assert not judged.inverse.any()
    assert (judged.dropped, judged.condition) == (3, 0.0)
    assert residuals(np.zeros((3, 3)), judged, RCOND) == (0.0, 0.0, 0.0)


def test_a_subspace_that_does_not_converge_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allowed no step, the referee refuses and returns no inverse."""
    monkeypatch.setattr("tests.decimal_inverse.MAX_STEPS", 0)
    with pytest.raises(RefereeError, match="did not converge"):
        referee(_path(4), RCOND)


def test_values_that_cross_the_cutoff_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LAPACK's values and the refined ones on different sides of the cutoff."""
    eigh = np.linalg.eigh

    def spoilt(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, vectors = eigh(matrix)
        values[0] = 0.0
        return values, vectors

    monkeypatch.setattr(np.linalg, "eigh", spoilt)
    with pytest.raises(RefereeError, match="cross the cutoff"):
        referee(np.diag([1e-3, 1.0, 2.0]), RCOND)
