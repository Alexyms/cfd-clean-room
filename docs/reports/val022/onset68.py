"""Prompt 68c, part 2: an estimate of the onset of shedding from two growth rates.

Near the onset of shedding the growth rate of the wake's oscillation passes
through zero roughly linearly in Re (a Hopf bifurcation). The straight line
through the growth rate at Re 40 (decaying, negative) and at Re 60 (growing,
positive) crosses zero at an estimate of Re_cr, reported beside Sohankar,
Norberg and Davidson's 51.2 +/- 1 (Table V). Both Reynolds numbers are about 10
from the onset, where the linear relation is only approximate. Reported, not
scored, except for prediction (h): the estimate lies between 45 and 57.

The readings, defined before they are taken (prompt 68c):

- Re 40: the probe's late decay rate, floor68.json ``probe_rate`` (the slope of
  ln(peak) in time over the peaks from t = 210, per floor68.peaks). This is the
  rate already in the records. The run-out's rate (runout68.json
  ``run_rate``, peaks from t = 250 before the first below 1e-6) gives the same
  estimate a second time when that record is present.
- Re 60: the envelope's growth rate in its early, small-amplitude phase, read
  from the row's lift series (val022_re60.json). The envelope is the half
  difference between each local maximum of C_L and the next local minimum (and
  each minimum and the next maximum), taken at the pair's mid time. It does not
  need the lift to change sign, which a steady offset left by the start-up's
  asymmetry could prevent while the amplitude is small.
- The window. A_sat is the median of the envelope over the last 50 time units.
  The window starts at the first envelope value after the envelope's minimum
  (over the run's first half; the hand-over from the start-up's decay to the
  growing mode) that reaches FLOOR * A_sat, and ends with the last value before
  the envelope first exceeds CEILING * A_sat. The rate is the least-squares
  slope of ln(envelope) in time over the window.
- Why it is linear. In the Landau amplitude equation dA/dt = sigma A (1 -
  A^2/A_sat^2) the growth rate of ln A is sigma times (1 - A^2/A_sat^2), so
  below CEILING = 0.1 of the saturated amplitude it is within 1% of the linear
  rate sigma. The floor keeps the fit away from the minimum, where a decaying
  start-up component still contributes. Whether ln(envelope) is straight in
  the window is read from the fit's largest residual and the local rates
  between successive values, both in the record.
- Sensitivity, reported: the window's other ends (CEILING 0.05 and 0.2, FLOOR
  0.03), and the Re 60 rate by floor68.peaks (the sign-change peaks) where the
  lift changes sign in the window.
- The periods. The oscillation's period in each window, twice the mean spacing
  of successive extrema, reported because a Hopf line compares one mode's rate
  at two Reynolds numbers.
- The estimate. The zero of the line through (40, rate_40) and (60, rate_60):
  Re_cr = 40 - rate_40 * 20 / (rate_60 - rate_40).

    python onset68.py

Writes onset68.json beside this script.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from floor68 import peaks, rate  # noqa: E402

PROBE_RECORD = HERE / "floor68.json"
RUNOUT_RECORD = HERE / "runout68.json"
ROW_RECORD = HERE / "val022_re60.json"
RECORD = HERE / "onset68.json"
REYNOLDS = (40.0, 60.0)
SOHANKAR = (51.2, 1.0)
PREDICTED_BAND = (45.0, 57.0)
FLOOR = 0.01
CEILING = 0.1
SATURATION_UNITS = 50.0
MINIMUM_VALUES = 4
SENSITIVITY = (
    ("primary", FLOOR, CEILING),
    ("ceiling 0.05", FLOOR, 0.05),
    ("ceiling 0.2", FLOOR, 0.2),
    ("floor 0.03", 0.03, CEILING),
)


def envelope(
    t: np.ndarray, c_l: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Half the difference between successive extrema of C_L, at the pair's mid time.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The mid times, the half differences, and the spacing of each pair.
    """
    step = np.diff(c_l)
    turn = np.nonzero(step[:-1] * step[1:] < 0.0)[0] + 1
    mid = 0.5 * (t[turn[:-1]] + t[turn[1:]])
    half = 0.5 * np.abs(c_l[turn[:-1]] - c_l[turn[1:]])
    return mid, half, np.diff(t[turn])


def growth(
    t: np.ndarray, c_l: np.ndarray, floor: float, ceiling: float
) -> dict[str, object]:
    """The envelope's growth rate over the window FLOOR to CEILING of the saturated amplitude.

    Returns
    -------
    dict[str, object]
        The rate per unit time, the window's times, its values' count, the
        fit's largest residual in ln(envelope), the local rates, the period,
        and the saturated amplitude; the rate is None when the window holds
        fewer than MINIMUM_VALUES values.
    """
    mid, half, spacing = envelope(t, c_l)
    saturated = float(np.median(half[mid >= mid[-1] - SATURATION_UNITS]))
    first_half = np.nonzero(mid <= 0.5 * mid[-1])[0]
    lowest = int(first_half[np.argmin(half[first_half])])
    reaches = np.nonzero(half[lowest:] >= floor * saturated)[0]
    result: dict[str, object] = {"saturated": saturated, "lowest_index": lowest}
    if reaches.size == 0:
        return {**result, "rate": None, "values": 0}
    start = lowest + int(reaches[0])
    over = np.nonzero(half[start:] > ceiling * saturated)[0]
    stop = start + (int(over[0]) if over.size else half.size - start)
    window = slice(start, stop)
    if stop - start < MINIMUM_VALUES:
        return {**result, "rate": None, "values": stop - start}
    slope, intercept = np.polyfit(mid[window], np.log(half[window]), 1)
    residual = np.log(half[window]) - (slope * mid[window] + intercept)
    local = np.diff(np.log(half[window])) / np.diff(mid[window])
    return {
        **result,
        "rate": float(slope),
        "values": stop - start,
        "from_t": float(mid[start]),
        "to_t": float(mid[stop - 1]),
        "from_amplitude": float(half[start]),
        "to_amplitude": float(half[stop - 1]),
        "largest_residual": float(np.max(np.abs(residual))),
        "local_rates": local.tolist(),
        "period": float(2.0 * np.mean(spacing[window])),
    }


def line_zero(rate_40: float, rate_60: float) -> float:
    """The Reynolds number where the line through the two rates crosses zero."""
    return REYNOLDS[0] - rate_40 * (REYNOLDS[1] - REYNOLDS[0]) / (rate_60 - rate_40)


def main() -> None:
    """Read the rates, form the estimate and write the record."""
    probe = json.loads(PROBE_RECORD.read_text(encoding="utf-8"))
    row = json.loads(ROW_RECORD.read_text(encoding="utf-8"))
    t = np.array(row["series"]["t"])
    c_l = np.array(row["series"]["c_l"])
    rate_40 = float(probe["probe_rate"])
    readings = {name: growth(t, c_l, lo, hi) for name, lo, hi in SENSITIVITY}
    primary = readings["primary"]
    estimates = {}
    for name, reading in readings.items():
        if reading["rate"] is not None:
            estimates[name] = line_zero(rate_40, float(reading["rate"]))
    estimate = estimates.get("primary")

    peak_t, peak_v = peaks(t, c_l)
    saturated = float(primary["saturated"])
    inside = (peak_t >= float(primary.get("from_t", 0.0))) & (
        peak_t <= float(primary.get("to_t", -1.0))
    )
    sign_change_rate = rate(peak_t[inside], peak_v[inside]) if inside.any() else None

    probe_times = np.array(probe["probe_peaks"]["t"])
    late = probe_times >= probe["from_t"] + probe["settle"]
    period_40 = float(2.0 * np.mean(np.diff(probe_times[late])))

    runout_estimate = None
    runout_rate = None
    if RUNOUT_RECORD.exists():
        runout = json.loads(RUNOUT_RECORD.read_text(encoding="utf-8"))
        runout_rate = runout["run_rate"]
        if runout_rate is not None and primary["rate"] is not None:
            runout_estimate = line_zero(float(runout_rate), float(primary["rate"]))

    record = {
        "kind": "onset",
        "reynolds": list(REYNOLDS),
        "rate_40": rate_40,
        "rate_40_source": "floor68.json probe_rate",
        "period_40": period_40,
        "runout_rate_40": runout_rate,
        "re60_readings": readings,
        "re60_sign_change_rate": sign_change_rate,
        "saturated": saturated,
        "estimate": estimate,
        "estimate_sensitivity": estimates,
        "estimate_with_runout_rate": runout_estimate,
        "sohankar": list(SOHANKAR),
        "predicted_band": list(PREDICTED_BAND),
        "prediction_h": (
            None
            if estimate is None
            else bool(PREDICTED_BAND[0] <= estimate <= PREDICTED_BAND[1])
        ),
    }
    RECORD.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
