from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .core import homogeneous_theorem_rate


@dataclass(frozen=True)
class FullHeterogeneityConfig:
    """Controlled n=2 EF21 configuration with heterogeneous L_i and mu_i."""

    epsilon: float
    tau_L: float
    tau_mu: float
    L_bar: float = 1.0
    mu_bar: float = 0.1

    def validate(self) -> None:
        if not (0.0 < self.epsilon < 1.0):
            raise ValueError("epsilon must lie in (0, 1)")
        if not (0.0 < self.tau_L <= 1.0):
            raise ValueError("tau_L must lie in (0, 1]")
        if not (0.0 < self.tau_mu <= 1.0):
            raise ValueError("tau_mu must lie in (0, 1]")
        if self.L_bar <= 0.0 or self.mu_bar <= 0.0:
            raise ValueError("L_bar and mu_bar must be positive")
        if self.mu_bar > self.L_bar:
            raise ValueError("mu_bar cannot exceed L_bar")

        L1, L2, mu1, mu2 = regularity_parameters(self)
        tol = 1e-14
        if mu1 > L1 + tol or mu2 > L2 + tol:
            raise ValueError(
                "controlled ratios violate strong-convexity/smoothness regularity: "
                f"require mu_i <= L_i, got (L1,mu1)=({L1},{mu1}) and "
                f"(L2,mu2)=({L2},{mu2})"
            )


def fixed_average_pair(tau: float, mean: float) -> tuple[float, float]:
    """Return x1,x2 with x2/x1=tau and (x1+x2)/2=mean."""
    if not (0.0 < tau <= 1.0):
        raise ValueError("tau must lie in (0, 1]")
    if mean <= 0.0:
        raise ValueError("mean must be positive")
    x1 = 2.0 * mean / (1.0 + tau)
    x2 = tau * x1
    return x1, x2


def regularity_parameters(config: FullHeterogeneityConfig) -> tuple[float, float, float, float]:
    L1, L2 = fixed_average_pair(config.tau_L, config.L_bar)
    mu1, mu2 = fixed_average_pair(config.tau_mu, config.mu_bar)
    return L1, L2, mu1, mu2


def empirical_eta_star_full(config: FullHeterogeneityConfig) -> float:
    """Reproduced Empirical Law 4.3 step size on the controlled full path."""
    config.validate()
    s = math.sqrt(config.epsilon)
    L1, L2, mu1, mu2 = regularity_parameters(config)
    denominator = (L1 + mu1) + (L2 + mu2)
    return 4.0 / denominator * (1.0 - s) / (1.0 + s)


def cubic_coefficients_full(config: FullHeterogeneityConfig) -> np.ndarray:
    """Empirical Law 4.3 cubic coefficients for heterogeneous L_i and mu_i."""
    config.validate()
    s = math.sqrt(config.epsilon)
    r = (1.0 - s) ** 2 / (1.0 + s)
    L1, L2, mu1, mu2 = regularity_parameters(config)
    Ls = np.array([L1, L2], dtype=float)
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


def optimal_contraction_factor_full(
    config: FullHeterogeneityConfig, tol: float = 1e-8
) -> float:
    """Return the largest real root of the reproduced empirical cubic."""
    coeffs = cubic_coefficients_full(config)
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


def analyze_full_config(config: FullHeterogeneityConfig) -> dict[str, float]:
    config.validate()
    L1, L2, mu1, mu2 = regularity_parameters(config)
    rho = optimal_contraction_factor_full(config)
    _, rho_h = homogeneous_theorem_rate(config.epsilon, config.L_bar, config.mu_bar)
    penalty = rho - rho_h
    normalized_penalty = penalty / max(1.0 - rho_h, 1e-15)
    return {
        "epsilon": config.epsilon,
        "tau_L": config.tau_L,
        "tau_mu": config.tau_mu,
        "L_bar": config.L_bar,
        "mu_bar": config.mu_bar,
        "kappa_bar": config.L_bar / config.mu_bar,
        "L1": L1,
        "L2": L2,
        "mu1": mu1,
        "mu2": mu2,
        "regularity_margin": min(L1 - mu1, L2 - mu2),
        "eta_star": empirical_eta_star_full(config),
        "rho_star": rho,
        "rho_homogeneous": rho_h,
        "heterogeneity_penalty": penalty,
        "normalized_penalty": normalized_penalty,
        "retention": 1.0 - normalized_penalty,
        "log_alignment_gap": abs(math.log(config.tau_L) - math.log(config.tau_mu)),
    }


def run_full_grid(
    tau_Ls: Iterable[float],
    tau_mus: Iterable[float],
    epsilons: Iterable[float],
    *,
    L_bar: float = 1.0,
    mu_bar: float = 0.1,
    skip_invalid: bool = True,
) -> tuple[list[dict[str, float]], int]:
    """Evaluate the controlled full-heterogeneity grid."""
    rows: list[dict[str, float]] = []
    invalid = 0
    for tau_L in tau_Ls:
        for tau_mu in tau_mus:
            for epsilon in epsilons:
                config = FullHeterogeneityConfig(
                    epsilon=epsilon,
                    tau_L=tau_L,
                    tau_mu=tau_mu,
                    L_bar=L_bar,
                    mu_bar=mu_bar,
                )
                try:
                    rows.append(analyze_full_config(config))
                except ValueError:
                    if not skip_invalid:
                        raise
                    invalid += 1
    return rows, invalid
