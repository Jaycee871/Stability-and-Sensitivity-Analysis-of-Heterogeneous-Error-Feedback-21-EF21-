from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TwoAgentConfig:
    epsilon: float
    tau: float
    L: float = 1.0
    mu_bar: float = 0.1

    def validate(self) -> None:
        if not (0.0 < self.epsilon < 1.0):
            raise ValueError("epsilon must lie in (0, 1)")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must lie in (0, 1]")
        if self.L <= 0.0 or self.mu_bar <= 0.0:
            raise ValueError("L and mu_bar must be positive")
        if self.mu_bar > self.L:
            raise ValueError("mu_bar cannot exceed L")
        mu1, _ = fixed_average_mus(self.tau, self.mu_bar)
        if mu1 > self.L + 1e-14:
            raise ValueError(
                "chosen tau and mu_bar imply mu1 > L; increase tau or reduce mu_bar"
            )


def fixed_average_mus(tau: float, mu_bar: float) -> tuple[float, float]:
    """Parameterize heterogeneity while holding average strong convexity fixed."""
    if not (0.0 < tau <= 1.0):
        raise ValueError("tau must lie in (0, 1]")
    if mu_bar <= 0.0:
        raise ValueError("mu_bar must be positive")
    mu1 = 2.0 * mu_bar / (1.0 + tau)
    mu2 = tau * mu1
    return mu1, mu2


def empirical_eta_star(config: TwoAgentConfig) -> float:
    """Empirical-law step size for n=2 on the controlled heterogeneity path."""
    config.validate()
    s = math.sqrt(config.epsilon)
    mu1, mu2 = fixed_average_mus(config.tau, config.mu_bar)
    denominator = 2.0 * config.L + mu1 + mu2
    return 4.0 / denominator * (1.0 - s) / (1.0 + s)


def cubic_coefficients(config: TwoAgentConfig) -> np.ndarray:
    """Return the reproduced Empirical Law 4.3 cubic coefficients."""
    config.validate()
    s = math.sqrt(config.epsilon)
    r = (1.0 - s) ** 2 / (1.0 + s)
    mu1, mu2 = fixed_average_mus(config.tau, config.mu_bar)
    Ls = np.array([config.L, config.L], dtype=float)
    mus = np.array([mu1, mu2], dtype=float)
    sigma = Ls + mus
    delta = Ls - mus
    k1 = (
        delta[1] ** 2 * sigma[0] + delta[0] ** 2 * sigma[1]
    ) / (sigma[0] * sigma[1] * sigma.sum())
    k2 = delta.sum() ** 2 / sigma.sum() ** 2
    return np.array(
        [
            1.0,
            -(s * (2.0 + s) + r * (s * k1 + k2)),
            s**2 * (1.0 + 2.0 * s + r * (k1 + s * k2)),
            -(s**4),
        ],
        dtype=float,
    )


def optimal_contraction_factor(config: TwoAgentConfig, tol: float = 1e-8) -> float:
    """Return the largest real root of the reproduced empirical cubic."""
    coeffs = cubic_coefficients(config)
    roots = np.roots(coeffs)
    real = roots[np.abs(roots.imag) <= tol].real
    if len(real) == 0:
        raise RuntimeError(f"no real root for config={config!r}; roots={roots!r}")

    rho = float(np.max(real))
    if not (-tol <= rho <= 1.0 + tol):
        raise RuntimeError(
            "largest real empirical-cubic root lies outside the contraction interval: "
            f"config={config!r}; largest_real_root={rho!r}; roots={roots!r}"
        )
    return rho


def homogeneous_theorem_rate(epsilon: float, L: float, mu: float) -> tuple[float, float]:
    """Theorem 3.1 step size and contraction rate for homogeneous parameters."""
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must lie in (0, 1)")
    if not (0.0 < mu <= L):
        raise ValueError("require 0 < mu <= L")
    s = math.sqrt(epsilon)
    eta = 2.0 / (L + mu) * (1.0 - s) / (1.0 + s)
    if math.isclose(L, mu, rel_tol=0.0, abs_tol=1e-14):
        return eta, s
    kappa = L / mu
    psi = 1.0 - s + math.sqrt(
        (1.0 + s) ** 2 + s * 16.0 * kappa / (kappa - 1.0) ** 2
    )
    rho = s + (1.0 - s) / 2.0 * ((kappa - 1.0) / (kappa + 1.0)) ** 2 * psi
    return eta, rho


def analyze_config(config: TwoAgentConfig) -> dict[str, float]:
    config.validate()
    mu1, mu2 = fixed_average_mus(config.tau, config.mu_bar)
    rho = optimal_contraction_factor(config)
    _, rho_h = homogeneous_theorem_rate(config.epsilon, config.L, config.mu_bar)
    penalty = rho - rho_h
    normalized_penalty = penalty / max(1.0 - rho_h, 1e-15)
    return {
        "epsilon": config.epsilon,
        "tau": config.tau,
        "L": config.L,
        "mu_bar": config.mu_bar,
        "mu1": mu1,
        "mu2": mu2,
        "eta_star": empirical_eta_star(config),
        "rho_star": rho,
        "rho_homogeneous": rho_h,
        "heterogeneity_penalty": penalty,
        "normalized_penalty": normalized_penalty,
    }


def run_grid(
    taus: Iterable[float],
    epsilons: Iterable[float],
    *,
    L: float = 1.0,
    mu_bar: float = 0.1,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for tau in taus:
        for epsilon in epsilons:
            rows.append(analyze_config(TwoAgentConfig(epsilon, tau, L=L, mu_bar=mu_bar)))
    return rows
