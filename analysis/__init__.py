# analysis/__init__.py
"""
Theoretical analysis and offline diagnostics tools:
- bellman_error: Value error, exact LSTD, MSPBE, and alignment conditions
- ntk: Empirical Neural Tangent Kernel (eNTK) and feature Jacobians
- policy_metrics: Policy and stationary distribution divergence metrics (TV, entropy)
"""

from analysis.ntk import compute_feature_jacobian, compute_eNTK, compute_centered_eNTK
from analysis.policy_metrics import compute_policy_metrics
from analysis.bellman_error import (
    Bellman_Residual_Exact,
    LSTD_Exact,
    LeastSquaresValue,
    get_error_vectors,
    compute_greedy_policy,
    evaluate_greedy_policy,
    value_metrics,
    value_metrics_light,
)

__all__ = [
    "compute_feature_jacobian",
    "compute_eNTK",
    "compute_centered_eNTK",
    "compute_policy_metrics",
    "Bellman_Residual_Exact",
    "LSTD_Exact",
    "LeastSquaresValue",
    "get_error_vectors",
    "compute_greedy_policy",
    "evaluate_greedy_policy",
    "value_metrics",
    "value_metrics_light",
]
