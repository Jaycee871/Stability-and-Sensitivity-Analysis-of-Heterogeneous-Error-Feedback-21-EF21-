from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ef21_stability import (  # noqa: E402
    FullHeterogeneityConfig,
    TwoAgentConfig,
    analyze_config,
    invariant_k2,
    mismatch_gap_closed_form,
    regularity_shape_coordinates,
    weighted_shape_variance,
)


def equal_smoothness_key_values() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for kappa_bar in (2.0, 10.0, 100.0):
        config = TwoAgentConfig(
            epsilon=0.95,
            tau=0.05,
            L=1.0,
            mu_bar=1.0 / kappa_bar,
        )
        result = analyze_config(config)
        rows.append(
            {
                "kappa_bar": kappa_bar,
                "epsilon": 0.95,
                "tau": 0.05,
                "rho_star": result["rho_star"],
                "rho_homogeneous": result["rho_homogeneous"],
                "normalized_penalty": result["normalized_penalty"],
            }
        )
    return rows


def mismatch_identity_checks() -> list[dict[str, float]]:
    checks: list[dict[str, float]] = []
    examples = [
        (2.0, 1.0, 0.05),
        (10.0, 0.05, 0.905),
        (100.0, 0.05, 1.0),
        (10.0, 0.4, 0.4),
    ]
    for kappa_bar, tau_L, tau_mu in examples:
        config = FullHeterogeneityConfig(
            epsilon=0.95,
            tau_L=tau_L,
            tau_mu=tau_mu,
            L_bar=1.0,
            mu_bar=1.0 / kappa_bar,
        )
        k1, k2 = regularity_shape_coordinates(config)
        closed = mismatch_gap_closed_form(tau_L, tau_mu, kappa_bar)
        variance = weighted_shape_variance(config)
        checks.append(
            {
                "kappa_bar": kappa_bar,
                "tau_L": tau_L,
                "tau_mu": tau_mu,
                "K1": k1,
                "K2": k2,
                "K2_closed": invariant_k2(kappa_bar),
                "K1_minus_K2": k1 - k2,
                "closed_form_gap": closed,
                "weighted_variance": variance,
                "identity_error": max(abs((k1 - k2) - closed), abs((k1 - k2) - variance)),
            }
        )
    return checks


def main() -> None:
    output = {
        "equal_smoothness_key_values": equal_smoothness_key_values(),
        "mismatch_identity_checks": mismatch_identity_checks(),
        "note": (
            "These calculations reproduce selected manuscript-facing values from the "
            "two-agent Empirical Law 4.3 analysis. Full grid summaries are stored in data/."
        ),
    }
    print(json.dumps(output, indent=2))

    max_error = max(row["identity_error"] for row in output["mismatch_identity_checks"])
    if not np.isfinite(max_error) or max_error > 1e-12:
        raise SystemExit(f"mismatch identity check failed: max error={max_error}")


if __name__ == "__main__":
    main()
