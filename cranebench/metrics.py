"""The frozen metric module.

Every number reported by the benchmark is produced by :func:`compute_metrics`
and by nothing else.  The module hashes its own source at import time and the
hash is written into every result file, so a reviewer can tell at a glance
whether two campaigns were scored by the same code.  If you change a metric,
the hash changes and old results stop comparing equal -- that is the intended
behaviour, not an inconvenience.

Definitions (all from the standardised outputs of :meth:`Plant.outputs`):

``ise_pos``        integral of squared horizontal position error [m^2 s]
``settle_time``    first time after which |pos error| stays below ``tol`` [s]
``peak_swing``     maximum |swing| over the run [deg]
``rms_swing``      root mean square swing [deg]
``residual_swing`` RMS swing over the last ``tail`` seconds [deg]
``peak_yaw``       maximum |yaw| [deg]
``rms_yaw``        RMS yaw [deg]
``effort``         integral of u^T u over the horizontal channels [N^2 s].
                   The hoist channel is excluded by construction: it carries the
                   static weight, so including it equalises every controller.
``peak_input``     max |u| over the horizontal channels [N]
``chatter``        integral of |du/dt| over the horizontal channels [N]
``final_pos_err``  mean |position error| over the last ``tail`` seconds [m].
                   Settling time is censored whenever the disturbance keeps the
                   payload out of the tolerance band, which under wind is
                   almost always; this metric still says whether the load
                   arrived. Without it a controller can score well by declining
                   to perform the transfer, which is exactly what a shaper
                   tuned far from its design frequency does.
``bound_ok``       1.0 if ``peak_swing`` never exceeds ``swing_bound``

``chatter`` is reported because a boundary-layer sliding controller can buy a
low ``ise_pos`` with a command that no drive will accept; effort alone does not
reveal that, and command roughness is a documented route to exciting modes that
the controller does not model.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np

METRIC_HASH = hashlib.sha256(
    pathlib.Path(__file__).read_bytes()).hexdigest()[:16]

DEG = 180.0 / np.pi


@dataclass
class Metrics:
    ise_pos: float
    settle_time: float
    peak_swing: float
    rms_swing: float
    residual_swing: float
    peak_yaw: float
    rms_yaw: float
    effort: float
    peak_input: float
    chatter: float
    final_pos_err: float
    bound_ok: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def compute_metrics(t: np.ndarray, outs: List[Dict[str, float]],
                    u: np.ndarray, ref_pos: np.ndarray,
                    horizontal_inputs=(0,),
                    swing_bound_deg: float = 4.8,
                    tol: float = 0.02, tail: float = 5.0) -> Metrics:
    t = np.asarray(t, float)
    u = np.atleast_2d(np.asarray(u, float))
    if u.shape[0] != t.size:
        u = u.T
    swing = np.array([o["swing"] for o in outs])
    yaw = np.array([o.get("yaw", 0.0) for o in outs])
    cart = np.array([o["cart"] for o in outs])
    err = cart - np.asarray(ref_pos, float)

    # Horizontal channels only.  The hoist channel carries the static weight of
    # the payload, so including it makes every controller's effort equal to
    # (m g)^2 T to three digits and destroys the comparison; the plant declares
    # which channels are horizontal drives.
    idx = [i for i in horizontal_inputs if i < u.shape[1]] or [0]
    horiz = u[:, idx]

    inside = np.abs(err) < tol
    if inside.all():
        settle = float(t[0])
    elif not inside.any():
        settle = float(t[-1])
    else:
        last_bad = int(np.where(~inside)[0][-1])
        settle = float(t[min(last_bad + 1, t.size - 1)])

    tail_mask = t >= (t[-1] - tail)
    peak_swing = float(np.max(np.abs(swing)) * DEG)

    return Metrics(
        ise_pos=_trapz(err ** 2, t),
        settle_time=settle,
        peak_swing=peak_swing,
        rms_swing=float(np.sqrt(np.mean(swing ** 2)) * DEG),
        residual_swing=float(np.sqrt(np.mean(swing[tail_mask] ** 2)) * DEG),
        peak_yaw=float(np.max(np.abs(yaw)) * DEG),
        rms_yaw=float(np.sqrt(np.mean(yaw ** 2)) * DEG),
        effort=_trapz(np.sum(horiz ** 2, axis=1), t),
        peak_input=float(np.max(np.abs(horiz))),
        chatter=float(np.sum(np.abs(np.diff(horiz, axis=0)))) if horiz.shape[0] > 1 else 0.0,
        final_pos_err=float(np.mean(np.abs(err[tail_mask]))),
        bound_ok=float(peak_swing <= swing_bound_deg),
    )
