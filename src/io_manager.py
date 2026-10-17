"""The run file's steady checkpoint, written and read back (ADR-014 D, step 0).

ADR-014 section D gives every run one HDF5 file and this module its writer.
Step 0 builds only the part the steady checkpoint needs: the
``/steady_checkpoint/`` group, holding a ``SteadyCheckpoint`` so that a
steady solve stopped at its cap continues, in another process, to what one
uninterrupted solve with the larger cap reaches, bit for bit. Step 3
inherits the group's layout and these two functions and adds the rest of
the file around them: ``/config`` with the merged configuration and its
hash (which a resume should then match before it continues), the flow
states, the transport frames, the march's ``/checkpoint/`` and the
periodic rewrite every ``checkpoint_interval`` outer iterations.

The group's layout, which step 3 keeps:

- datasets ``u`` [ny, nx+1], ``v`` [ny+1, nx], ``p`` [ny, nx], and with the
  turbulence section ``k``, ``eps``, ``nu_t`` [ny, nx], all float64;
- datasets ``residual_history``, ``velocity_steps`` and ``viscosity_steps``,
  float64, one dimension, possibly empty;
- attributes ``outer`` and ``pressure_cap_hits`` (integers), and
  ``stop_reason`` and ``rule_version`` when not None.

HDF5 stores float64 arrays as their bytes, so what is read back is what was
written, bitwise. Nothing here reads a configuration or decides anything
about a solve; the solver validates a checkpoint when it resumes from one.
"""

from pathlib import Path

import h5py
import numpy as np

from src.solver_staggered import FlowState, SteadyCheckpoint

STEADY_CHECKPOINT = "steady_checkpoint"
_FIELDS = ("u", "v", "p")
_TURBULENCE = ("k", "eps", "nu_t")
_SERIES = ("residual_history", "velocity_steps", "viscosity_steps")


def write_steady_checkpoint(path: str | Path, checkpoint: SteadyCheckpoint) -> None:
    """Write a steady checkpoint into a run file, replacing any already there.

    Parameters
    ----------
    path : str or Path
        The run file. Created if absent; its other groups are kept.
    checkpoint : SteadyCheckpoint
        ``StaggeredSolver.steady_checkpoint`` of the solve to continue.

    Raises
    ------
    TypeError
        If ``checkpoint`` is not a SteadyCheckpoint.
    """
    if not isinstance(checkpoint, SteadyCheckpoint):
        raise TypeError(
            f"checkpoint must be a SteadyCheckpoint, got {type(checkpoint).__name__}"
        )
    state = checkpoint.state
    with h5py.File(path, "a") as run_file:
        if STEADY_CHECKPOINT in run_file:
            del run_file[STEADY_CHECKPOINT]
        group = run_file.create_group(STEADY_CHECKPOINT)
        names = _FIELDS + (_TURBULENCE if state.turbulent else ())
        for name in names:
            group.create_dataset(name, data=getattr(state, name), dtype="<f8")
        for name in _SERIES:
            values = np.asarray(getattr(checkpoint, name), dtype=np.float64)
            group.create_dataset(name, data=values, dtype="<f8")
        group.attrs["outer"] = state.outer
        group.attrs["pressure_cap_hits"] = checkpoint.pressure_cap_hits
        if state.stop_reason is not None:
            group.attrs["stop_reason"] = state.stop_reason
        if checkpoint.rule_version is not None:
            group.attrs["rule_version"] = checkpoint.rule_version


def read_steady_checkpoint(path: str | Path) -> SteadyCheckpoint:
    """Read back the steady checkpoint ``write_steady_checkpoint`` wrote.

    Parameters
    ----------
    path : str or Path
        The run file.

    Returns
    -------
    SteadyCheckpoint
        The checkpoint as written, its arrays and series bitwise.

    Raises
    ------
    KeyError
        If the file holds no ``/steady_checkpoint/`` group.
    """
    with h5py.File(path, "r") as run_file:
        if STEADY_CHECKPOINT not in run_file:
            raise KeyError(f"{path} holds no /{STEADY_CHECKPOINT}/ group")
        group = run_file[STEADY_CHECKPOINT]
        arrays = {name: group[name][()] for name in _FIELDS}
        if all(name in group for name in _TURBULENCE):
            arrays |= {name: group[name][()] for name in _TURBULENCE}
        series = {name: tuple(float(x) for x in group[name][()]) for name in _SERIES}
        attrs = dict(group.attrs)
    reason = attrs.get("stop_reason")
    rule_version = attrs.get("rule_version")
    state = FlowState(
        **arrays,
        stop_reason=None if reason is None else str(reason),
        outer=int(attrs["outer"]),
    )
    return SteadyCheckpoint(
        state=state,
        residual_history=series["residual_history"],
        velocity_steps=series["velocity_steps"],
        viscosity_steps=series["viscosity_steps"],
        rule_version=None if rule_version is None else int(rule_version),
        pressure_cap_hits=int(attrs["pressure_cap_hits"]),
    )
