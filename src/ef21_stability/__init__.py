"""Controlled stability and sensitivity analysis for heterogeneous EF21."""

from .core import (
    TwoAgentConfig,
    analyze_config,
    cubic_coefficients,
    empirical_eta_star,
    fixed_average_mus,
    homogeneous_theorem_rate,
    optimal_contraction_factor,
    run_grid,
)
from .full_heterogeneity import (
    FullHeterogeneityConfig,
    analyze_full_config,
    cubic_coefficients_full,
    empirical_eta_star_full,
    fixed_average_pair,
    optimal_contraction_factor_full,
    regularity_parameters,
    run_full_grid,
)
from .structure import (
    admissible_tau_mu_bounds,
    cubic_discriminant_full,
    invariant_k2,
    k1_root_sensitivity,
    mismatch_gap_closed_form,
    regularity_shape_coordinates,
    weighted_shape_variance,
)

__all__ = [
    "TwoAgentConfig",
    "analyze_config",
    "cubic_coefficients",
    "empirical_eta_star",
    "fixed_average_mus",
    "homogeneous_theorem_rate",
    "optimal_contraction_factor",
    "run_grid",
    "FullHeterogeneityConfig",
    "analyze_full_config",
    "cubic_coefficients_full",
    "empirical_eta_star_full",
    "fixed_average_pair",
    "optimal_contraction_factor_full",
    "regularity_parameters",
    "run_full_grid",
    "regularity_shape_coordinates",
    "invariant_k2",
    "mismatch_gap_closed_form",
    "weighted_shape_variance",
    "admissible_tau_mu_bounds",
    "cubic_discriminant_full",
    "k1_root_sensitivity",
]
