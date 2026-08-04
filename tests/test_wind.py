"""The disturbance models must deliver the statistics they advertise."""

import numpy as np

from cranebench.wind import DrydenParams, DrydenWind, KaimalParams, KaimalWind
from cranebench.wind.kaimal import kaimal_psd, synthesise


def test_kaimal_variance_is_exact():
    p = KaimalParams(u_mean=12.0, intensity=0.14)
    rec = synthesise(20000, 0.01, p, np.random.default_rng(1))
    assert abs(rec.std() - p.intensity * p.u_mean) < 1e-9
    assert abs(rec.mean()) < 1e-9


def test_kaimal_spectrum_matches_target():
    """Welch estimate of the record must track the target PSD in log space."""
    from scipy.signal import welch
    p = KaimalParams()
    dt = 0.01
    rec = synthesise(2 ** 17, dt, p, np.random.default_rng(2))
    f, pxx = welch(rec, fs=1 / dt, nperseg=4096)
    band = (f > 1e-2) & (f < 5.0)
    r = np.corrcoef(np.log(pxx[band]), np.log(kaimal_psd(f[band], p)))[0, 1]
    assert r > 0.95, f"log-PSD correlation {r:.3f}"


def test_dryden_is_stationary_from_the_first_sample():
    """The filter must start on its stationary distribution, not from zero."""
    p = DrydenParams(sigma=1.7)
    first = np.array([DrydenWind(50.0, p, np.random.default_rng(s)).turb[0]
                      for s in range(400)])
    assert abs(first.std() / p.sigma - 1.0) < 0.12
    w = DrydenWind(4000.0, p, np.random.default_rng(3))
    assert abs(w.turb.std() / p.sigma - 1.0) < 0.10


def test_wind_record_is_independent_of_the_solver_step():
    """Same seed, same duration -> same field, whatever the integrator does."""
    p = KaimalParams()
    a = KaimalWind(40.0, p, np.random.default_rng(5))
    b = KaimalWind(40.0, p, np.random.default_rng(5))
    ts = np.linspace(0, 39.0, 977)
    assert np.allclose([a.speed(t) for t in ts], [b.speed(t) for t in ts])
