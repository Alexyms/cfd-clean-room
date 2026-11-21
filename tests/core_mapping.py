"""The mapping between the compiled core's plain data and the reference's types (ADR-017 B).

The binding is handed data and returns data: a class's budget as its ten terms
by name, and a breach as ``_cfdcore.ConservationBreach`` carrying the class, the
leftover, the particles supplied and the limit. The reference's types for the
same things are ``MassBudget`` and ``ConservationBreachError``. Step 6's
``src/core_backend.py`` will hold this mapping; until it exists the parity tests
and prompt 90's probe share it here, so that every comparison is made on the
reference's own types and the binding imports nothing from ``src``.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.solver_transport import DEPOSIT_SURFACES, ConservationBreachError, MassBudget

if TYPE_CHECKING:
    import _cfdcore


def budget_from_terms(terms: Mapping[str, float | None]) -> MassBudget:
    """A MassBudget from the ten terms the core returns by name.

    Parameters
    ----------
    terms : Mapping[str, float | None]
        ``initial``, ``inflow``, ``outflow``, ``source``, one term per name of
        ``DEPOSIT_SURFACES``, ``current`` and ``leftover``. ``initial`` and
        ``current`` are None before the first step.

    Returns
    -------
    MassBudget
        A new budget holding the same values.
    """
    return MassBudget(
        initial=terms["initial"],
        inflow=terms["inflow"],
        outflow=terms["outflow"],
        source=terms["source"],
        deposited={surface: terms[surface] for surface in DEPOSIT_SURFACES},
        current=terms["current"],
        leftover=terms["leftover"],
    )


def terms_of(budget: MassBudget) -> dict[str, float | None]:
    """A MassBudget's ten terms by name, as the core takes them.

    Parameters
    ----------
    budget : MassBudget
        A budget whose ``deposited`` keys are ``DEPOSIT_SURFACES``.

    Returns
    -------
    dict[str, float | None]
        ``initial``, ``inflow``, ``outflow``, ``source``, the four deposition
        terms by surface name, ``current`` and ``leftover``.
    """
    return {
        "initial": budget.initial,
        "inflow": budget.inflow,
        "outflow": budget.outflow,
        "source": budget.source,
        **{surface: budget.deposited[surface] for surface in DEPOSIT_SURFACES},
        "current": budget.current,
        "leftover": budget.leftover,
    }


def budgets_of(core: "_cfdcore.TransportSolver2") -> list[MassBudget]:
    """Every class's budget of a core solver as new MassBudget objects."""
    return [budget_from_terms(terms) for terms in core.budgets()]


def restore_budgets(
    core: "_cfdcore.TransportSolver2", budgets: Sequence[MassBudget]
) -> None:
    """Set a core solver's budgets whole from MassBudget objects.

    The refusals and their messages are ``TransportSolver.restore_budgets``'s:
    they concern ``MassBudget`` and a checkpoint's leftover, which the core knows
    nothing of. The terms are then handed to the core as plain data.

    Parameters
    ----------
    core : _cfdcore.TransportSolver2
        The solver.
    budgets : Sequence[MassBudget]
        One per class, in class order.

    Raises
    ------
    TypeError
        If an element is not a MassBudget.
    ValueError
        If there is not one per class, a budget's ``deposited`` keys are not the
        four deposition surfaces, or its leftover is unknown.
    """
    if len(budgets) != core.classes:
        raise ValueError(
            f"restore_budgets needs one budget per class ({core.classes}), "
            f"got {len(budgets)}"
        )
    for k, budget in enumerate(budgets):
        if not isinstance(budget, MassBudget):
            raise TypeError(
                f"budget {k} must be a MassBudget, got {type(budget).__name__}"
            )
        if tuple(budget.deposited) != DEPOSIT_SURFACES:
            raise ValueError(
                f"budget {k} books deposition to {list(budget.deposited)}, not "
                f"{list(DEPOSIT_SURFACES)}"
            )
        if budget.leftover is None:
            raise ValueError(
                f"budget {k}'s leftover is unknown: it was written before the "
                "implicit solve's leftover was booked (prompt 78), so the "
                "accumulated leftover before the restart is unknown and REQ-T05's "
                "bound cannot be carried across it (ADR-016)"
            )
    core.restore_budgets([terms_of(budget) for budget in budgets])


def step(
    core: "_cfdcore.TransportSolver2", size_class: int, c: np.ndarray, out: np.ndarray
) -> "_cfdcore.TransportStep2":
    """One class step of a core solver, a breach raised as the reference's error.

    Parameters
    ----------
    core : _cfdcore.TransportSolver2
        The solver, its interval and step set.
    size_class : int
        The class.
    c : np.ndarray
        The class field, [ny, nx].
    out : np.ndarray
        The field advanced, [ny, nx].

    Returns
    -------
    _cfdcore.TransportStep2
        The step's records.

    Raises
    ------
    ConservationBreachError
        When the core raises ``_cfdcore.ConservationBreach``: the class's
        accumulated leftover reached the limit (ADR-016).
    """
    # Imported here, so that this module loads where the core is not built.
    import _cfdcore

    try:
        return core.step(size_class, c, out)
    except _cfdcore.ConservationBreach as breach:
        raise ConservationBreachError(
            breach.size_class, breach.leftover, breach.supplied, breach.limit
        ) from breach
