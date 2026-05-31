"""Bergman 1979 FSIGT healthy-cohort canonical parameter table.

Reference (model + cohort calibration):

    Bergman, R. N., Ider, Y. Z., Bowden, C. R., & Cobelli, C. (1979).
    Quantitative estimation of insulin sensitivity. *American Journal of
    Physiology* 236, E667-E677.

The Bergman minimal model parameters were originally calibrated on a
healthy adult cohort using the *frequently-sampled intravenous glucose
tolerance test* (FSIGT). The "canonical healthy" parameter values
quoted in textbooks and reproduced in nearly every minimal-model study
since 1979 are tabulated here for paper-parity benchmarking of the
``glucose_insulin`` test model.

These are not in /min units (the original publication's convention) but
in /hour units (this repo's convention). The unit conversions are noted
inline.

Numerical values are uncopyrightable as facts; the table itself is a
re-derivation from the published mean values for healthy subjects.
"""

from __future__ import annotations

from typing import Final


# Bergman 1979 healthy-cohort means (converted to /hour units).
PARAMS: Final[dict] = {
    'p1':           1.8,           # /hr   (= 0.030/min, glucose effectiveness)
    'p2':           1.5,           # /hr   (= 0.025/min, remote insulin decay)
    'p3':           4.68e-2,       # /(hr²·μU/mL) (= 1.3e-5 /(min²·μU/mL))
    'k':            18.0,          # /hr   (= 0.30/min, plasma insulin clearance)
    'Gb':           90.0,          # mg/dL (basal glucose)
    'Ib':           7.0,           # μU/mL (basal insulin)
    'V_G':          1.6,           # dL/kg (glucose volume of distribution)
    'V_I':          1.2,           # dL/kg (insulin volume of distribution)
    'BW':           70.0,          # kg    (representative adult body weight)
    'n_beta':       8.0,           # /(hr·μU/mL/(mg/dL)) — Bergman 1981 ext.
    'h_beta':       90.0,          # mg/dL (β-cell secretion threshold)
}


# Insulin sensitivity index — the canonical Bergman 1979 inference target.
# SI = p₃ / p₂ in /(hr·μU/mL); equivalently in /(min·μU/mL) × 60.
SI: Final[float] = PARAMS['p3'] / PARAMS['p2']     # ≈ 0.0312 /(hr·μU/mL)


# Expected qualitative response to a 40g mixed-meal challenge in a
# healthy adult parameterised at the Bergman 1979 cohort means
# (validated against the simulator on commit 2026-04-28):
EXPECTED_HEALTHY_MEAL_RESPONSE: Final[dict] = {
    'peak_G_mg_dL':              (165, 200),     # post-meal peak range
    'peak_t_post_meal_hr':       (0.5, 1.0),     # peak occurs 30-60 min after meal
    'return_to_basal_hr':        (1.5, 2.5),     # back to Gb within 1.5-2.5 hr
    'peak_I_mu_U_mL':            (40, 60),        # postprandial insulin peak
}


REFERENCE: Final[str] = (
    "Bergman, R. N., Ider, Y. Z., Bowden, C. R., & Cobelli, C. (1979). "
    "Quantitative estimation of insulin sensitivity. American Journal of "
    "Physiology 236, E667-E677. The Bergman 1981 β-cell-secretion "
    "extension and Bergman 1989 review fix the canonical /hour-unit "
    "parameter values reproduced here."
)


__all__ = ['PARAMS', 'SI', 'EXPECTED_HEALTHY_MEAL_RESPONSE', 'REFERENCE']
