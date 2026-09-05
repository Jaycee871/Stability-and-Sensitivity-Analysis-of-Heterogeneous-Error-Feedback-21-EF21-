# Stability and Sensitivity Analysis of Heterogeneous Error Feedback 21 (EF21)

Curated public reproducibility repository for the manuscript **“Stability and Sensitivity Analysis of Heterogeneous Error Feedback 21 (EF21) under Communication Compression.”**

Authors: Pack Kwan Low and Fu-Hsing Wang, Department of Information Management, Chinese Culture University.

## Scope

This repository contains the public, manuscript-facing subset of the computational study: derived numerical summaries, symbolic certificates, and minimal code needed to reproduce the reported EF21 stability and sensitivity analyses.

The private research workspace is intentionally not mirrored here. Internal research logs, exploratory notes, novelty-audit materials, OSF workflow files, model/tool interaction records, and unpublished working artifacts are excluded.

## Repository layout

- `data/` — curated derived results used by the manuscript.
- `src/ef21_stability/` — minimal numerical implementation of the reproduced two-agent Empirical Law 4.3 and the fixed-average heterogeneity analyses.
- `requirements.txt` — Python dependencies.
- `manuscript/` — reserved for the public LaTeX/Overleaf-facing manuscript package.

## Scientific scope

The study reproduces the two-agent heterogeneous EF21 Empirical Law 4.3 and analyzes its contraction prediction under controlled communication compression and regularity heterogeneity. The public package includes the equal-smoothness baseline, contraction-margin retention results, robustness checks, the full-regularity mismatch factorization, and the generic cubic root/sensitivity certificate.

All conclusions concerning the predicted contraction factor remain conditional on the reproduced Empirical Law 4.3; this repository does not claim a new general EF21 convergence theorem.

## Reproducibility

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Archival citation/DOI information will be added when the public release is frozen.
