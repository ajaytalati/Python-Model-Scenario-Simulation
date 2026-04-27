"""Anderson & May 1978 boarding-school influenza outbreak.

The canonical paper-parity benchmark for stochastic SIR inference. A
single influenza A (H1N1) outbreak in a boys' boarding school of 763
students, January 1978; tracked daily by the resident school physician
who reported "number confined to bed" each morning.

Originally published in:

    Anonymous (1978). Influenza in a boarding school. *British Medical
    Journal* 1, 587. (Outbreak report; data table.)

Re-tabulated and analysed in:

    Anderson, R. M., & May, R. M. (1991). *Infectious Diseases of
    Humans: Dynamics and Control*. Oxford University Press.
    Table 6.1 (p. 124, daily-prevalence series).

Used as the canonical particle-MCMC tutorial benchmark in:

    Endo, A., van Leeuwen, E., & Baguelin, M. (2019). Introduction to
    particle Markov-chain Monte Carlo for disease dynamics modellers.
    *Epidemics* 29, 100363. https://doi.org/10.1016/j.epidem.2019.100363

Reference truth parameters from those analyses (frequency-dependent SIR):

    N      = 763
    β      = 1.66 / day  (transmission rate)
    γ      = 0.5  / day  (recovery rate; mean infectious period 2 days)
    R₀     = β / γ ≈ 3.32
    I_0    = 1   (single index case introduced day 0)

The 14-day daily-prevalence series (number confined to bed each day) is
the de-facto fixture used for state-space-filter benchmarks. The total
attack rate ~512/763 = 67% (sum across days here is double-counts since
prevalence is not incidence; the standard convention is to report the
final R(t) ≈ 720, attack rate ≈ 94%).

This is a numerical fixture; the data themselves carry no copyright (a
14-element integer series is uncopyrightable as a fact). Numerical
values are cross-verified against the Endo et al 2019 supplementary R
code (publicly available on github.com/akira-endo/PMCMC_tutorial).
"""

from __future__ import annotations

from typing import Final


# Daily prevalence — boys confined to bed at the morning roll-call,
# day 1 (outbreak start) through day 14 (outbreak end).
# Values from Anderson & May 1991 Table 6.1.
DAILY_PREVALENCE: Final[tuple[int, ...]] = (
    3,    # Day 1
    8,    # Day 2
    26,   # Day 3
    76,   # Day 4
    225,  # Day 5
    298,  # Day 6
    258,  # Day 7
    233,  # Day 8
    189,  # Day 9
    128,  # Day 10
    68,   # Day 11
    29,   # Day 12
    14,   # Day 13
    4,    # Day 14
)

POPULATION: Final[int] = 763
DAYS: Final[int] = len(DAILY_PREVALENCE)

REFERENCE: Final[str] = (
    "Anderson & May 1991, Table 6.1 (boarding-school flu, BMJ 1978). "
    "Used as canonical PMCMC benchmark in Endo, van Leeuwen, Baguelin "
    "(2019) Epidemics 29, https://doi.org/10.1016/j.epidem.2019.100363."
)


# Convenience structured fixture (the bundle most consumers will want)
ANDERSON_MAY_1978_FLU = {
    'name':              'anderson_may_1978_flu',
    'reference':         REFERENCE,
    'population':        POPULATION,
    'days':              DAYS,
    'daily_prevalence':  DAILY_PREVALENCE,
    'truth_params': {
        'N':            POPULATION,
        'beta_per_day': 1.66,
        'gamma_per_day': 0.5,
        'R_0':          3.32,
    },
    'truth_init': {
        'I_0': 1,
    },
}


def peak_prevalence() -> int:
    """The peak daily prevalence (≈ peak I_t)."""
    return max(DAILY_PREVALENCE)


def peak_day() -> int:
    """The day index (1-indexed) on which prevalence peaks."""
    return DAILY_PREVALENCE.index(peak_prevalence()) + 1
