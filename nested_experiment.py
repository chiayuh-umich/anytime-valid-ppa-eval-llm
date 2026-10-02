"""Standalone nested confidence-sequence experiments.

FAQ predictor attribution: initialization and sequential Laplace updates are
adapted from Skyler Wu et al.'s efficiently-evaluating-llms repository:
https://github.com/skbwu/efficiently-evaluating-llms (faq_final.py).
The fitted v, mu=mean(U), and sigma=cov(U.T) are supplied in predictor.npz.
This runner uses float64 and its existing [1e-6, 1-1e-6] probability clipping.
See LICENSE-FAQ for the upstream Apache-2.0 license.

RIPr, testing-by-betting, Hedged-CS and Hedged-WoR; uniform / max-min /
max-max querying; FAQ, controlled oracle and custom predictors.
All numerical solvers are defined here. See README.md for reproduction commands.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ============================================================================
# RIPr projection and FAQ posterior update
# ============================================================================

DEFAULT_EPSILONS = (0.05, 0.075, 0.1, 0.125, 0.15)


RIPR_PROB_EPS = 1e-6


RIPR_Q_EPS = 1e-12


RIPR_BRACKET_STEPS = 24


RIPR_BISECTION_STEPS = 28


def bernoulli_kl(p, q):
    p = p.clamp(min=RIPR_PROB_EPS, max=1.0 - RIPR_PROB_EPS)
    q = q.clamp(min=RIPR_PROB_EPS, max=1.0 - RIPR_PROB_EPS)
    return (
        p * (torch.log(p) - torch.log(q))
        + ((1.0 - p) * (torch.log1p(-p) - torch.log1p(-q)))
    )


def compute_ripr_point_given_dual_lambda(reference_probs_js, q_js, dual_lambda_values):
    if reference_probs_js.dim() == 1:
        reference_probs_js = reference_probs_js.unsqueeze(0)
    if q_js.dim() == 1:
        q_js = q_js.unsqueeze(0)

    reference_probs_js = reference_probs_js.to(dtype=torch.float64).clamp(
        min=RIPR_PROB_EPS,
        max=1.0 - RIPR_PROB_EPS,
    )
    q_js = q_js.to(dtype=torch.float64).clamp_min(RIPR_Q_EPS)
    dual_lambda_values = torch.as_tensor(
        dual_lambda_values,
        dtype=torch.float64,
        device=reference_probs_js.device,
    )

    if dual_lambda_values.dim() == 0:
        dual_lambda_values = dual_lambda_values.view(1, 1)
    elif dual_lambda_values.dim() == 1:
        if reference_probs_js.shape[0] == 1:
            dual_lambda_values = dual_lambda_values.view(1, -1)
        elif dual_lambda_values.numel() == reference_probs_js.shape[0]:
            dual_lambda_values = dual_lambda_values.view(-1, 1)
        else:
            raise ValueError("dual_lambda_values has incompatible shape.")
    elif dual_lambda_values.dim() != 2:
        raise ValueError("dual_lambda_values must be scalar, vector, or matrix.")

    batch_size, n_questions = reference_probs_js.shape
    n_targets = dual_lambda_values.shape[1]
    if dual_lambda_values.shape[0] != batch_size:
        if batch_size == 1:
            reference_probs_js = reference_probs_js.expand(dual_lambda_values.shape[0], -1)
            q_js = q_js.expand(dual_lambda_values.shape[0], -1)
            batch_size = dual_lambda_values.shape[0]
        else:
            raise ValueError("dual_lambda_values first dimension must match batch size.")

    reference_expanded = reference_probs_js.unsqueeze(1).expand(batch_size, n_targets, n_questions)
    q_expanded = q_js.unsqueeze(1).expand(batch_size, n_targets, n_questions)
    dual_lambda_expanded = dual_lambda_values.unsqueeze(-1)

    base_term = dual_lambda_expanded + q_expanded
    discriminant = (base_term ** 2) - (
        4.0 * dual_lambda_expanded * q_expanded * reference_expanded
    )
    sqrt_discriminant = torch.sqrt(discriminant.clamp_min(0.0))
    projected_probs = (2.0 * q_expanded * reference_expanded) / (
        base_term + sqrt_discriminant
    ).clamp_min(RIPR_Q_EPS)
    projected_probs = projected_probs.clamp(min=RIPR_PROB_EPS, max=1.0 - RIPR_PROB_EPS)
    return projected_probs[:, 0, :] if n_targets == 1 else projected_probs


def solve_ripr_dual_lambda_batch(reference_probs_js, q_js, target_means):
    if reference_probs_js.dim() == 1:
        reference_probs_js = reference_probs_js.unsqueeze(0)
    if q_js.dim() == 1:
        q_js = q_js.unsqueeze(0)

    target_means = torch.as_tensor(
        target_means,
        dtype=torch.float64,
        device=reference_probs_js.device,
    ).clamp(min=RIPR_PROB_EPS, max=1.0 - RIPR_PROB_EPS)
    if target_means.dim() == 0:
        target_means = target_means.view(1, 1)
    elif target_means.dim() == 1:
        if reference_probs_js.shape[0] == 1:
            target_means = target_means.view(1, -1)
        elif target_means.numel() == reference_probs_js.shape[0]:
            target_means = target_means.view(-1, 1)
        else:
            raise ValueError("target_means has incompatible shape.")
    elif target_means.dim() != 2:
        raise ValueError("target_means must be scalar, vector, or matrix.")

    if target_means.shape[0] != reference_probs_js.shape[0] and reference_probs_js.shape[0] != 1:
        raise ValueError("target_means first dimension must match batch size.")

    lambda_lo = torch.full_like(target_means, fill_value=-1.0)
    lambda_hi = torch.full_like(target_means, fill_value=1.0)

    def mean_under_dual_lambda(lambda_batch):
        projected_probs = compute_ripr_point_given_dual_lambda(
            reference_probs_js=reference_probs_js,
            q_js=q_js,
            dual_lambda_values=lambda_batch,
        )
        if projected_probs.dim() == 2:
            return projected_probs.mean(dim=1, keepdim=True)
        return projected_probs.mean(dim=2)

    for _ in range(RIPR_BRACKET_STEPS):
        mean_lo = mean_under_dual_lambda(lambda_lo)
        mean_hi = mean_under_dual_lambda(lambda_hi)
        need_lower = mean_lo < target_means
        need_upper = mean_hi > target_means
        if (not need_lower.any()) and (not need_upper.any()):
            break
        lambda_lo = torch.where(need_lower, 2.0 * lambda_lo, lambda_lo)
        lambda_hi = torch.where(need_upper, 2.0 * lambda_hi, lambda_hi)

    for _ in range(RIPR_BISECTION_STEPS):
        lambda_mid = 0.5 * (lambda_lo + lambda_hi)
        mean_mid = mean_under_dual_lambda(lambda_mid)
        move_lower_up = mean_mid > target_means
        lambda_lo = torch.where(move_lower_up, lambda_mid, lambda_lo)
        lambda_hi = torch.where(move_lower_up, lambda_hi, lambda_mid)

    return 0.5 * (lambda_lo + lambda_hi)


def compute_ripr_endpoint_stats(reference_probs_js, q_js, endpoint_targets):
    q64 = q_js.to(dtype=torch.float64).clamp_min(RIPR_Q_EPS)
    reference64 = reference_probs_js.to(dtype=torch.float64).clamp(
        min=RIPR_PROB_EPS,
        max=1.0 - RIPR_PROB_EPS,
    )
    dual_lambdas = solve_ripr_dual_lambda_batch(
        reference_probs_js=reference64,
        q_js=q64,
        target_means=endpoint_targets,
    )
    projected_probs = compute_ripr_point_given_dual_lambda(
        reference_probs_js=reference64,
        q_js=q64,
        dual_lambda_values=dual_lambdas,
    )

    if projected_probs.dim() == 2:
        kl_terms = bernoulli_kl(reference64, projected_probs)
        growth = (q64 * kl_terms).sum(dim=1)
        log_if_one = torch.log(reference64) - torch.log(projected_probs)
        log_if_zero = torch.log1p(-reference64) - torch.log1p(-projected_probs)
        conditional_means = (
            q64 * (reference64 * log_if_one + (1.0 - reference64) * log_if_zero)
        ).sum(dim=1)
        conditional_second_moments = (
            q64
            * (reference64 * (log_if_one ** 2) + (1.0 - reference64) * (log_if_zero ** 2))
        ).sum(dim=1)
    else:
        reference_expanded = reference64.unsqueeze(1).expand_as(projected_probs)
        q_expanded = q64.unsqueeze(1).expand_as(projected_probs)
        kl_terms = bernoulli_kl(reference_expanded, projected_probs)
        growth = (q_expanded * kl_terms).sum(dim=2)
        log_if_one = torch.log(reference_expanded) - torch.log(projected_probs)
        log_if_zero = torch.log1p(-reference_expanded) - torch.log1p(-projected_probs)
        conditional_means = (
            q_expanded
            * (
                reference_expanded * log_if_one
                + (1.0 - reference_expanded) * log_if_zero
            )
        ).sum(dim=2)
        conditional_second_moments = (
            q_expanded
            * (
                reference_expanded * (log_if_one ** 2)
                + (1.0 - reference_expanded) * (log_if_zero ** 2)
            )
        ).sum(dim=2)

    variance = (conditional_second_moments - (conditional_means ** 2)).clamp_min(0.0)
    return projected_probs, growth, variance, kl_terms


def prepare_interval_endpoints(current_lbs, current_ubs):
    empty_cs = current_lbs > current_ubs
    endpoint_lbs = torch.minimum(current_lbs, current_ubs)
    endpoint_ubs = torch.maximum(current_lbs, current_ubs)
    endpoint_lbs = torch.where(empty_cs, torch.zeros_like(endpoint_lbs), endpoint_lbs)
    endpoint_ubs = torch.where(empty_cs, torch.ones_like(endpoint_ubs), endpoint_ubs)
    return endpoint_lbs, endpoint_ubs


def update_factor_posterior(uhats, sigmahats, p1mp_hat_js, v, sampled_indices, sampled_labels, sampled_phats):
    """FAQ covariance/mean update, adapted from upstream faq_final.py, Part 3.

    Update covariance first; the mean step uses that updated covariance.
    Only the current queried response and the pre-query probability enter.
    """
    w_s = torch.gather(input=p1mp_hat_js, dim=1, index=sampled_indices)
    v_is = v[sampled_indices.flatten()]
    sigma_v_is = torch.bmm(sigmahats, v_is.unsqueeze(-1))
    v_t_sigma_v_is = torch.bmm(v_is.unsqueeze(-2), sigma_v_is).squeeze(-1)
    denominator = 1.0 + (w_s * v_t_sigma_v_is)
    numerator = w_s.unsqueeze(-1) * (sigma_v_is @ sigma_v_is.mT)
    sigmahats = sigmahats - (numerator / denominator.unsqueeze(-1))
    uhats = uhats + torch.bmm(
        sigmahats,
        ((sampled_labels - sampled_phats) * v_is).unsqueeze(-1),
    ).squeeze(-1)
    return uhats, sigmahats


# ============================================================================
# Exact predictable betting fractions
# ============================================================================

BET_Q_EPS = 1e-12


BET_FACTOR_EPS = 1e-12


BET_LAMBDA_BISECTION_STEPS = 40


def coerce_target_matrix(target_values, batch_size, device):
    target_values = torch.as_tensor(
        target_values,
        dtype=torch.float64,
        device=device,
    )
    if target_values.dim() == 0:
        return target_values.view(1, 1).expand(batch_size, 1)
    if target_values.dim() == 1:
        if batch_size == 1:
            return target_values.view(1, -1)
        if target_values.numel() == batch_size:
            return target_values.view(batch_size, 1)
        raise ValueError("target_values has incompatible shape.")
    if target_values.dim() == 2:
        if target_values.shape[0] == batch_size:
            return target_values
        if target_values.shape[0] == 1:
            return target_values.expand(batch_size, -1)
        raise ValueError("target_values first dimension must match batch size.")
    raise ValueError("target_values must be scalar, vector, or matrix.")


def coerce_h_vector(h_values, batch_size, device):
    h_values = torch.as_tensor(
        h_values,
        dtype=torch.float64,
        device=device,
    )
    if h_values.dim() == 0:
        return h_values.view(1).expand(batch_size)
    if h_values.dim() == 1:
        if h_values.numel() == 1:
            return h_values.expand(batch_size)
        if h_values.numel() == batch_size:
            return h_values
        raise ValueError("h_values has incompatible shape.")
    raise ValueError("h_values must be scalar or vector.")


def compute_betting_lambda_bounds(target_means, h_values, n_questions):
    target_means = target_means.to(dtype=torch.float64)
    h_values = h_values.to(dtype=torch.float64).view(-1, 1).clamp_min(BET_Q_EPS)
    inv_nh = 1.0 / (float(n_questions) * h_values)
    lower = -1.0 / (((1.0 - target_means) + inv_nh).clamp_min(BET_Q_EPS))
    upper = 1.0 / ((target_means + inv_nh).clamp_min(BET_Q_EPS))
    return lower, upper


def compute_betting_expected_log_growths_and_lambda_derivatives_batch(
    reference_probs_js,
    q_js,
    target_means,
    bet_lambdas,
):
    if reference_probs_js.dim() == 1:
        reference_probs_js = reference_probs_js.unsqueeze(0)
    if q_js.dim() == 1:
        q_js = q_js.unsqueeze(0)

    batch_size, n_questions = reference_probs_js.shape
    device = reference_probs_js.device
    target_matrix = coerce_target_matrix(target_means, batch_size, device)
    lambda_matrix = coerce_target_matrix(bet_lambdas, batch_size, device)

    reference_probs = reference_probs_js.to(dtype=torch.float64).clamp(min=BET_Q_EPS, max=1.0 - BET_Q_EPS)
    q_values = q_js.to(dtype=torch.float64).clamp_min(BET_Q_EPS)
    theta_hats = reference_probs.mean(dim=1, keepdim=True)

    p_expanded = reference_probs.unsqueeze(1)
    q_expanded = q_values.unsqueeze(1)
    centered_means = theta_hats.unsqueeze(1) - target_matrix.unsqueeze(-1)
    delta_if_one = centered_means + ((1.0 - p_expanded) / (float(n_questions) * q_expanded))
    delta_if_zero = centered_means - (p_expanded / (float(n_questions) * q_expanded))

    lambda_expanded = lambda_matrix.unsqueeze(-1)
    factor_if_one = (1.0 + (lambda_expanded * delta_if_one)).clamp_min(BET_FACTOR_EPS)
    factor_if_zero = (1.0 + (lambda_expanded * delta_if_zero)).clamp_min(BET_FACTOR_EPS)

    growth_terms = q_expanded * (
        p_expanded * torch.log(factor_if_one)
        + (1.0 - p_expanded) * torch.log(factor_if_zero)
    )
    derivative_terms = q_expanded * (
        (p_expanded * delta_if_one / factor_if_one)
        + ((1.0 - p_expanded) * delta_if_zero / factor_if_zero)
    )
    growths = growth_terms.sum(dim=2)
    derivatives = derivative_terms.sum(dim=2)
    return growths, derivatives


def solve_optimal_betting_lambda_batch(reference_probs_js, q_js, target_means, h_values):
    if reference_probs_js.dim() == 1:
        reference_probs_js = reference_probs_js.unsqueeze(0)
    if q_js.dim() == 1:
        q_js = q_js.unsqueeze(0)

    batch_size = reference_probs_js.shape[0]
    device = reference_probs_js.device
    target_matrix = coerce_target_matrix(target_means, batch_size, device).clamp(
        min=BET_Q_EPS,
        max=1.0 - BET_Q_EPS,
    )
    h_vector = coerce_h_vector(h_values, batch_size, device).clamp_min(BET_Q_EPS)

    lower_bounds, upper_bounds = compute_betting_lambda_bounds(
        target_means=target_matrix,
        h_values=h_vector,
        n_questions=reference_probs_js.shape[1],
    )
    _, deriv_lower = compute_betting_expected_log_growths_and_lambda_derivatives_batch(
        reference_probs_js=reference_probs_js,
        q_js=q_js,
        target_means=target_matrix,
        bet_lambdas=lower_bounds,
    )
    _, deriv_upper = compute_betting_expected_log_growths_and_lambda_derivatives_batch(
        reference_probs_js=reference_probs_js,
        q_js=q_js,
        target_means=target_matrix,
        bet_lambdas=upper_bounds,
    )

    choose_lower = deriv_lower <= 0.0
    choose_upper = deriv_upper >= 0.0
    interior = ~(choose_lower | choose_upper)
    left = lower_bounds.clone()
    right = upper_bounds.clone()
    for _ in range(BET_LAMBDA_BISECTION_STEPS):
        if not interior.any():
            break
        mid = 0.5 * (left + right)
        _, deriv_mid = compute_betting_expected_log_growths_and_lambda_derivatives_batch(
            reference_probs_js=reference_probs_js,
            q_js=q_js,
            target_means=target_matrix,
            bet_lambdas=mid,
        )
        move_left_up = deriv_mid > 0.0
        left = torch.where(interior & move_left_up, mid, left)
        right = torch.where(interior & (~move_left_up), mid, right)

    optimal_lambdas = 0.5 * (left + right)
    optimal_lambdas = torch.where(choose_lower, lower_bounds, optimal_lambdas)
    optimal_lambdas = torch.where(choose_upper, upper_bounds, optimal_lambdas)
    return optimal_lambdas


def compute_betting_expected_log_growths_batch(reference_probs_js, q_js, target_means, bet_lambdas):
    growths, _ = compute_betting_expected_log_growths_and_lambda_derivatives_batch(
        reference_probs_js=reference_probs_js,
        q_js=q_js,
        target_means=target_means,
        bet_lambdas=bet_lambdas,
    )
    return growths


# ============================================================================
# Prepared data and reproducible random streams
# ============================================================================

ROOT = Path(__file__).resolve().parent


SCENARIOS = {
    # Preserve the historical random-stream IDs when renaming the datasets.
    "z_3": ("unpermuted", "high"),
    "tilde_z_3": ("permuted", "high"),
    "z_2": ("unpermuted", "medium"),
    "tilde_z_2": ("permuted", "medium"),
    "z_1": ("unpermuted", "similar"),
    "tilde_z_1": ("permuted", "similar"),
}

# Accept old run/data directories while writing new outputs with paper names.
LEGACY_SCENARIO_NAMES = {
    "small_similar": "z_1", "small_medium": "z_2", "small_high": "z_3",
    "large_similar": "tilde_z_1", "large_medium": "tilde_z_2", "large_high": "tilde_z_3",
}


def canonical_scenario(name):
    return LEGACY_SCENARIO_NAMES.get(name, name)


def read_data_manifest(path):
    manifest = json.loads(Path(path).read_text())
    for row in manifest.get("scenarios", []):
        row["scenario"] = canonical_scenario(row["scenario"])
    return manifest


def scenario_label(name):
    name = canonical_scenario(name)
    if name in SCENARIOS:
        symbol = r"\tilde{z}" if name.startswith("tilde_") else "z"
        return rf"${symbol}_{{{name.rsplit('_', 1)[1]}}}$"
    return name


PROB_EPS = RIPR_PROB_EPS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def empty_output_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Use a new directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Must be a nonnegative integer.")
    return value


def query_uniforms(seed, scenario_index, repeat_ids, max_steps):
    """Streams are stable under method order, repeat batching, and horizon extension."""
    return np.stack([
        np.random.default_rng(np.random.SeedSequence([seed, scenario_index, int(repeat)]))
        .random(max_steps) for repeat in repeat_ids
    ])


def predictions(uhats, v):
    return torch.sigmoid(uhats @ v.T).clamp(PROB_EPS, 1.0 - PROB_EPS)


def advance_predictor(uhats, sigmahats, p, v, indices, labels):
    return update_factor_posterior(
        uhats, sigmahats, p * (1.0 - p), v, indices, labels, p.gather(1, indices),
    )


def load_prepared(data_dir, device):
    data_dir = Path(data_dir)
    manifest = read_data_manifest(data_dir / "manifest.json")
    if manifest.get("schema_version") != 1 or not manifest.get("mismatch_validated"):
        raise ValueError("Expected a validated fixed-z benchmark manifest (schema 1).")
    snapshot = data_dir / manifest["predictor_file"]
    if sha256(snapshot) != manifest["predictor_sha256"]:
        raise ValueError("Predictor snapshot checksum mismatch.")
    with np.load(snapshot, allow_pickle=False) as arrays:
        v, mu, sigma = [torch.tensor(arrays[key], dtype=torch.float64, device=device) for key in ("v", "mu", "sigma")]
    if v.ndim != 2 or mu.shape != (v.shape[1],) or sigma.shape != (v.shape[1], v.shape[1]):
        raise ValueError("Incompatible predictor snapshot shapes.")
    if not all(torch.isfinite(x).all() for x in (v, mu, sigma)):
        raise ValueError("Nonfinite predictor snapshot.")
    records = {row["scenario"]: row for row in manifest["scenarios"]}
    original_scenarios = {name for name, (_, accuracy) in SCENARIOS.items() if accuracy != "similar"}
    if (len(manifest["scenarios"]) != len(records)
            or set(records) not in (original_scenarios, set(SCENARIOS))):
        raise ValueError("Manifest must contain z_2, z_3 and their tilde versions, optionally also z_1 and tilde_z_1.")
    zs = {}
    for scenario, row in records.items():
        path = data_dir / row["z_file"]
        if sha256(path) != row["z_sha256"]:
            raise ValueError(f"Fixed z checksum mismatch: {scenario}.")
        values = pd.read_csv(path).to_numpy(dtype=np.float64)
        if values.shape != (1, v.shape[0]) or not np.isin(values, [0, 1]).all():
            raise ValueError(f"Expected exactly one binary z with {v.shape[0]} questions: {path}.")
        if values.mean() != row["theta_star"]:
            raise ValueError(f"Realized accuracy disagrees with the manifest: {scenario}.")
        zs[scenario] = torch.tensor(values[0], dtype=torch.float64, device=device)
    return manifest, zs, v, mu, sigma


# ============================================================================
# Controlled oracle predictor
# ============================================================================

BENCHMARK_KIND = "fixed_z_oracle"


LEGACY_BENCHMARK_KIND = "mean_matched_fixed_z_oracle"


MISMATCH_METRICS = ("kl", "brier")


class OraclePredictionSequence:
    """Pre-query p at query s=t+1: (1-a_s) z + a_s c.

    KL control uses a_s=a0/s**beta; Brier uses a_s=a0/s**(beta/2).
    The fixed center c is in (0, 1), defaulting to mean(z) for legacy runs.
    All regimes share exactly the same initial predictor, set by a0, c and z.
    Mean prediction error is a_s * (c - mean(z)); mean matching is optional.

    Full z is intentional oracle side information. This is not a fitted learner.
    Refuse horizons at which the shared CS probability clipping changes the law.
    """

    def __init__(self, z, amplitude, decay_power, horizon, mismatch_metric="kl", oracle_center=None):
        if z.ndim != 1 or z.numel() < 2 or not ((z == 0) | (z == 1)).all():
            raise ValueError("Expected a binary z vector with at least two items.")
        self.z = z.detach().to(dtype=torch.float64).clone()
        self.theta = int(z.sum()) / z.numel()
        if not 0 < self.theta < 1:
            raise ValueError("Both labels must occur in z.")
        self.center = self.theta if oracle_center is None else float(oracle_center)
        if not math.isfinite(self.center) or not 0 < self.center < 1:
            raise ValueError("oracle center must be finite and strictly between 0 and 1.")
        if not math.isfinite(amplitude) or not 0 < amplitude <= 1:
            raise ValueError("oracle amplitude must be finite and in (0, 1].")
        if not math.isfinite(decay_power) or decay_power < 0:
            raise ValueError("decay powers must be finite and nonnegative.")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        if mismatch_metric not in MISMATCH_METRICS:
            raise ValueError(f"mismatch_metric must be one of {MISMATCH_METRICS}.")
        self.mismatch_metric = mismatch_metric
        self.mixing_decay_power = decay_power if mismatch_metric == "kl" else decay_power / 2
        self.amplitude, self.decay_power, self.horizon = amplitude, decay_power, horizon
        last_weight = amplitude * math.exp(-self.mixing_decay_power * math.log(horizon))
        if last_weight * min(self.center, 1 - self.center) < PROB_EPS:
            raise ValueError(
                "This schedule would reach the CS probability clipping floor. "
                "Increase --oracle-amplitude, move --oracle-center away from 0 and 1, "
                "or decrease --max-steps or --decay-powers. "
                "The runner will not silently replace the requested mismatch decay by a linear tail."
            )

    def __call__(self, t, n_repeats):
        if not 0 <= t < self.horizon:
            raise ValueError("Query step is outside the validated horizon.")
        weight = self.amplitude * (t + 1) ** (-self.mixing_decay_power)
        p = (1 - weight) * self.z + weight * self.center
        return p[None, :].expand(n_repeats, -1)

    def diagnostics(self):
        steps = np.arange(1, self.horizon + 1, dtype=np.float64)
        weights = self.amplitude * steps ** (-self.mixing_decay_power)
        kl_one = -np.log1p(-weights * (1 - self.center))
        kl_zero = -np.log1p(-weights * self.center)
        lower, upper = np.minimum(kl_one, kl_zero), np.maximum(kl_one, kl_zero)
        brier_one, brier_zero = (weights * (1 - self.center)) ** 2, (weights * self.center) ** 2
        brier_lower, brier_upper = np.minimum(brier_one, brier_zero), np.maximum(brier_one, brier_zero)
        uniform_kl = self.theta * kl_one + (1 - self.theta) * kl_zero
        uniform_brier = self.theta * brier_one + (1 - self.theta) * brier_zero
        mean_error = weights * (self.center - self.theta)
        return pd.DataFrame({
            "step": steps.astype(np.int64), "oracle_mixing_weight": weights,
            "oracle_center": self.center,
            "expected_prediction_mean": self.theta + mean_error,
            "expected_prediction_mean_error": mean_error,
            "kl_lower_bound": lower, "kl_upper_bound": upper,
            "cum_kl_lower_bound": np.cumsum(lower), "cum_kl_upper_bound": np.cumsum(upper),
            "brier_lower_bound": brier_lower, "brier_upper_bound": brier_upper,
            "cum_brier_lower_bound": np.cumsum(brier_lower), "cum_brier_upper_bound": np.cumsum(brier_upper),
            "expected_uniform_kl": uniform_kl, "cum_expected_uniform_kl": np.cumsum(uniform_kl),
            "expected_brier_z": uniform_brier, "cum_expected_brier_z": np.cumsum(uniform_brier),
        })


# ============================================================================
# Constrained max-min querying optimizer
# ============================================================================

OPT_RTOL = 1e-4


OPT_ATOL = 1e-9


MAX_BACKTRACKS = 12


def validate_uniform_weight(uniform_weight):
    value = float(uniform_weight)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("uniform_weight must be finite and in (0, 1].")
    return value


def with_uniform_floor(scores, uniform_weight):
    """Parameterize the constrained simplex as q = (1-eta) s + eta/N."""
    eta = validate_uniform_weight(uniform_weight)
    scores = scores.to(dtype=torch.float64)
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Query scores must be finite and nonnegative.")
    totals = scores.sum(dim=-1, keepdim=True)
    if (totals <= 0).any():
        raise ValueError("Query scores must have a positive sum.")
    return (1.0 - eta) * (scores / totals) + eta / scores.shape[-1]


def _endpoint_models(family, p, q, lower, upper, h):
    """Both endpoint values and their q derivatives, without choosing an active end."""
    targets = torch.stack([lower, upper], dim=1)
    if family == "ripr":
        _, growths, _, gradients = compute_ripr_endpoint_stats(p, q, targets)
    elif family == "bet":
        with torch.no_grad():
            lambdas = solve_optimal_betting_lambda_batch(p, q, targets, h)
        with torch.enable_grad():
            q_var = q.detach().clone().requires_grad_(True)
            values = compute_betting_expected_log_growths_batch(
                p, q_var, targets, lambdas.detach(),
            )
            gradients = torch.stack([
                torch.autograd.grad(values[:, 0].sum(), q_var, retain_graph=True)[0],
                torch.autograd.grad(values[:, 1].sum(), q_var)[0],
            ], dim=1)
            growths = values.detach()
    else:
        raise ValueError(f"Unknown CS family: {family}")
    if not torch.isfinite(growths).all() or not torch.isfinite(gradients).all():
        raise FloatingPointError("Nonfinite max-min objective or gradient.")
    return growths.detach(), gradients.detach()


def _mirror_trial(s, growths, gradients_s, step):
    """Entropic step for the minimum of the two affine endpoint models.

    A scalar dual weight balances the two models, so a change in the worse
    endpoint does not force us to follow an arbitrary one-sided derivative.
    This only constructs a proposal: acceptance always uses the actual min.
    """
    centered = gradients_s - (gradients_s * s[:, None, :]).sum(2, keepdim=True)
    active = growths.argmin(dim=1)
    rows = torch.arange(s.shape[0], device=s.device)
    scale = centered[rows, active].abs().amax(dim=1).clamp_min(1e-15)
    directions = centered / scale[:, None, None]
    offsets = (growths - growths.amin(dim=1, keepdim=True)) / scale[:, None]
    log_s = s.clamp_min(1e-300).log()
    difference = directions[:, 0] - directions[:, 1]

    def at(weight):
        direction = directions[:, 1] + weight[:, None] * difference
        proposal = torch.softmax(log_s + step[:, None] * direction, dim=1)
        derivative = offsets[:, 0] - offsets[:, 1] + ((proposal - s) * difference).sum(1)
        return proposal, derivative

    lo, hi = torch.zeros_like(step), torch.ones_like(step)
    at_zero, derivative_zero = at(lo)
    at_one, derivative_one = at(hi)
    interior = (derivative_zero < 0) & (derivative_one > 0)
    if interior.any():
        for _ in range(24):
            mid = (lo + hi) / 2
            _, derivative = at(mid)
            lo = torch.where(derivative < 0, mid, lo)
            hi = torch.where(derivative >= 0, mid, hi)
        balanced, _ = at((lo + hi) / 2)
    else:
        balanced = at_zero
    proposal = torch.where((derivative_zero >= 0)[:, None], at_zero, at_one)
    proposal = torch.where(interior[:, None], balanced, proposal).clamp_min(1e-300)
    proposal = proposal / proposal.sum(1, keepdim=True)
    linear_values = growths + (gradients_s * (proposal - s)[:, None, :]).sum(2)
    predicted_gain = linear_values.amin(1) - growths.amin(1)
    return proposal, predicted_gain


def maxmin_distribution(predictions, lower, upper, family, uniform_weight=0.05,
                        opt_steps=25, learning_rate=0.5, *, initial_q=None,
                        opt_rtol=OPT_RTOL, opt_atol=OPT_ATOL):
    """Maximize the same floored endpoint minimum with multiple numerical starts.

    Search independently from uniform, sqrt(p(1-p)), and optionally the previous
    query distribution. Scale mirror steps and backtrack on the actual min.
    Progress-based stopping is not a certificate of global optimality.
    """
    eta = validate_uniform_weight(uniform_weight)
    if int(opt_steps) != opt_steps or opt_steps < 0:
        raise ValueError("opt_steps must be a nonnegative integer.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    if any(not math.isfinite(x) or x < 0 for x in (opt_rtol, opt_atol)):
        raise ValueError("Optimizer tolerances must be finite and nonnegative.")
    p = predictions.detach().to(dtype=torch.float64)
    if p.ndim != 2 or p.shape[1] < 1:
        raise ValueError("predictions must have shape (repeats, questions).")
    if eta / p.shape[1] < max(BET_Q_EPS, RIPR_Q_EPS):
        raise ValueError("eta/N must be at least 1e-12 for the underlying numerical solvers.")
    if not torch.isfinite(p).all() or (p <= 0).any() or (p >= 1).any():
        raise ValueError("Predictions must be finite and strictly between 0 and 1.")
    lower = torch.as_tensor(lower, dtype=p.dtype, device=p.device)
    upper = torch.as_tensor(upper, dtype=p.dtype, device=p.device)
    if lower.shape != (p.shape[0],) or upper.shape != lower.shape:
        raise ValueError("CS endpoints must have one entry per repeat.")
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError("CS endpoints must be finite.")
    lower, upper = prepare_interval_endpoints(lower, upper)
    h = torch.full((p.shape[0],), eta / p.shape[1], dtype=p.dtype, device=p.device)
    uniform = torch.full_like(p, 1.0 / p.shape[1])
    starts = [uniform]
    if eta < 1:
        scores = torch.sqrt(p * (1.0 - p))
        starts.append(scores / scores.sum(1, keepdim=True))
    if initial_q is not None:
        initial_q = torch.as_tensor(initial_q, dtype=p.dtype, device=p.device).detach()
        if (initial_q.shape != p.shape or not torch.isfinite(initial_q).all()
                or (initial_q < h[:, None] - 1e-12).any()
                or not torch.allclose(initial_q.sum(1), torch.ones_like(h), atol=1e-12, rtol=1e-12)):
            raise ValueError("initial_q must be a normalized distribution satisfying the floor.")
        if eta < 1:
            scores = ((initial_q - h[:, None]) / (1.0 - eta)).clamp_min(1e-300)
            starts.append(scores / scores.sum(1, keepdim=True))

    repeats, n_starts = p.shape[0], len(starts)
    s = torch.cat(starts)
    pp, ll, uu, hh = p.repeat(n_starts, 1), lower.repeat(n_starts), upper.repeat(n_starts), h.repeat(n_starts)
    q = with_uniform_floor(s, eta)
    # Keep the uniform candidate bit-for-bit equal to the uniform policy.
    q[:repeats] = uniform
    growths, gradients = _endpoint_models(family, pp, q, ll, uu, hh)
    uniform_value = growths[:repeats].amin(1).clone()
    start_best_value = growths.amin(1).view(n_starts, repeats).amax(0)
    status = torch.zeros(s.shape[0], dtype=torch.long, device=p.device)
    iterations, evaluations = torch.zeros_like(status), torch.ones_like(status)
    step_sizes = torch.full_like(hh, learning_rate)
    running = torch.full_like(status, eta < 1, dtype=torch.bool)
    if eta == 1:
        status.fill_(3)

    for _ in range(int(opt_steps)):
        pending = running.nonzero().flatten()
        if not pending.numel():
            break
        iterations[pending] += 1
        for _ in range(MAX_BACKTRACKS):
            trial_s, predicted_gain = _mirror_trial(
                s[pending], growths[pending], (1.0 - eta) * gradients[pending], step_sizes[pending],
            )
            old_value = growths[pending].amin(1)
            tolerance = opt_atol + opt_rtol * old_value.abs()
            small = predicted_gain <= tolerance
            status[pending[small]] = 1
            running[pending[small]] = False
            pending, trial_s, predicted_gain = pending[~small], trial_s[~small], predicted_gain[~small]
            if not pending.numel():
                break
            trial_q = with_uniform_floor(trial_s, eta)
            trial_growths, trial_gradients = _endpoint_models(
                family, pp[pending], trial_q, ll[pending], uu[pending], hh[pending],
            )
            evaluations[pending] += 1
            gain = trial_growths.amin(1) - growths[pending].amin(1)
            accepted = (gain > 0) & (gain >= 1e-4 * predicted_gain)
            take = pending[accepted]
            s[take], q[take] = trial_s[accepted], trial_q[accepted]
            growths[take], gradients[take] = trial_growths[accepted], trial_gradients[accepted]
            step_sizes[take] = (step_sizes[take] * 2.0).clamp_max(max(64.0, learning_rate))
            pending = pending[~accepted]
            if not pending.numel():
                break
            step_sizes[pending] *= 0.5
        if pending.numel():
            status[pending] = 2
            running[pending] = False

    values = growths.amin(1).view(n_starts, repeats)
    winner = values.argmax(0)
    rows = winner * repeats + torch.arange(repeats, device=p.device)
    best_value = growths[rows].amin(1)
    return q[rows].detach(), h, {
        "growth_lower": growths[rows, 0], "growth_upper": growths[rows, 1],
        "min_growth": best_value, "uniform_candidate_growth": uniform_value,
        "initial_candidate_growth": start_best_value,
        "gain_over_uniform": best_value - uniform_value,
        "optimizer_iterations": iterations.view(n_starts, repeats).sum(0),
        "optimizer_evaluations": evaluations.view(n_starts, repeats).sum(0),
        "optimizer_status": status[rows], "optimizer_start": winner,
        "optimizer_budget_hit": running.view(n_starts, repeats).any(0),
    }


def sample_from_uniforms(q, uniforms):
    """Inverse-CDF sampling with one independent uniform variate per repeat."""
    if q.ndim != 2 or uniforms.shape != (q.shape[0], 1):
        raise ValueError("Expected q=(repeats, questions), uniforms=(repeats, 1).")
    if not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError("Invalid query probabilities.")
    if not torch.allclose(q.sum(dim=1), torch.ones_like(q[:, 0]), atol=1e-12, rtol=1e-12):
        raise ValueError("Query probabilities must sum to one.")
    if not torch.isfinite(uniforms).all() or (uniforms < 0).any() or (uniforms >= 1).any():
        raise ValueError("Uniform draws must lie in [0, 1).")
    cumulative = q.cumsum(dim=1)
    cumulative[:, -1] = 1.0
    return torch.searchsorted(cumulative.contiguous(), uniforms.contiguous(), right=True)


# ============================================================================
# Real M2 model preparation and loading
# ============================================================================

M2_BANK_KIND = "m2_model_bank_v1"


M2_METADATA = ["data_kind", "model_row", "model_name", "model_sha", "reference_row",
            "reference_theta_star", "accuracy_gap", "n_models_selected",
            "question_columns_sha256", "predictor_snapshot_sha256"]


def select_farthest(values, reference_row):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2
            or not 0 <= reference_row < values.shape[0] or not np.isin(values, [0, 1]).all()):
        raise ValueError("Need at least two M2 models, binary nonmissing labels, and a valid reference row")
    counts = values.sum(axis=1).astype(np.int64)
    accuracies = counts/values.shape[1]
    gaps = np.abs(counts-counts[reference_row])/values.shape[1]
    candidates = np.delete(np.arange(len(values)), reference_row)
    # candidates are in stored row order: ties resolve to the lowest row index.
    winner = int(candidates[np.argmax(gaps[candidates])])
    return winner, accuracies, gaps


def prepare_m2(args):
    """Compare accuracies on the same first N questions; preserve the chosen labels."""
    torch.set_num_threads(1)
    source = Path(args.m2_path)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    n = args.n_questions
    questions = header[3:3+n]
    if n < 2 or header[:3] != ["model", "created_date", "sha"] or questions != [str(i) for i in range(n)]:
        raise ValueError("M2 must have model/created_date/sha followed by ordered question columns 0..N-1")
    frame = pd.read_csv(source, usecols=header[:3]+questions)
    all_values = frame.loc[:, questions].to_numpy(dtype=np.float64)
    winner, accuracies, gaps = select_farthest(all_values, args.reference_row)
    rows = args.model_rows or [winner]
    if (len(set(rows)) != len(rows) or args.reference_row in rows
            or any(row < 0 or row >= len(frame) for row in rows)):
        raise ValueError("Choose distinct valid model rows excluding the reference row")
    values = all_values[rows]
    if frame.loc[rows, "model"].isna().any():
        raise ValueError("Selected model name is missing")
    snapshot = Path(args.predictor_snapshot)
    with np.load(snapshot, allow_pickle=False) as arrays:
        v, mu, sigma = [torch.as_tensor(arrays[key], dtype=torch.float64)
                        for key in ("v", "mu", "sigma")]
    if (v.ndim != 2 or v.shape[0] < n or mu.shape != (v.shape[1],)
            or sigma.shape != (v.shape[1], v.shape[1])
            or not all(torch.isfinite(x).all() for x in (v, mu, sigma))):
        raise ValueError("Predictor snapshot must contain finite v, mu, sigma for these questions")
    v = v[:n].clone()
    p = predictions(mu[None, :], v)[0].numpy()
    output = empty_output_directory(args.output_dir)
    np.savez_compressed(output/"predictor.npz", v=v.numpy(), mu=mu.numpy(), sigma=sigma.numpy())
    records = []
    for row, z in zip(rows, values):
        row = int(row)
        scenario = f"m2_row_{row:04d}"
        z_file = f"{scenario}_z.csv"
        pd.DataFrame([z.astype(int)], columns=questions).to_csv(output/z_file, index=False)
        record = dict(scenario=scenario, model_row=row, model_name=str(frame.at[row, "model"]),
                      model_sha=str(frame.at[row, "sha"]), created_date=str(frame.at[row, "created_date"]),
                      n_questions=n, n_correct=int(z.sum()), theta_star=float(z.mean()),
                      z_file=z_file, z_sha256=sha256(output/z_file),
                      accuracy_gap=float(gaps[row]),
                      initial_prediction_mean=float(p.mean()), initial_mae_z=float(np.abs(p-z).mean()),
                      initial_brier_z=float(np.square(p-z).mean()),
                      initial_kl_z=float(np.where(z == 1, -np.log(p), -np.log1p(-p)).mean()))
        records.append(record)
    pd.DataFrame(records).to_csv(output/"selected_models.csv", index=False)
    pd.DataFrame(dict(model_row=np.arange(len(frame)), model_name=frame.model,
                      theta_star=accuracies, absolute_accuracy_gap=gaps,
                      is_selected=np.isin(np.arange(len(frame)), rows),
                      is_reference=np.arange(len(frame)) == args.reference_row)).to_csv(
                          output/"model_accuracy_scan.csv", index=False)
    manifest = dict(benchmark_kind=M2_BANK_KIND, schema_version=1, base_dataset=args.base_dataset,
                    n_questions=n, n_models=len(rows),
                    selection=("Explicit zero-based M2 rows" if args.model_rows else
                               "Largest absolute accuracy difference; ties choose lowest M2 row"),
                    reference_row=args.reference_row, reference_model_name=str(frame.at[args.reference_row, "model"]),
                    reference_theta_star=float(accuracies[args.reference_row]), accuracy_gap=float(gaps[winner]),
                    excluded_rows=[args.reference_row], selected_rows=rows,
                    n_source_models=len(frame), m2_path=str(source.resolve()), m2_sha256=sha256(source),
                    question_columns=questions,
                    question_columns_sha256=hashlib.sha256(json.dumps(questions).encode()).hexdigest(),
                    construction="Unaltered binary answers of selected M2 models on the same first N stored questions",
                    coverage_target="per-model theta_star = sum(z_model)/N; queries and predictor states reset per repeat",
                    predictor_file="predictor.npz", predictor_sha256=sha256(output/"predictor.npz"),
                    source_predictor_sha256=sha256(snapshot),
                    generator_sha256=sha256(Path(__file__)), scenarios=records)
    write_json(output/"manifest.json", manifest)
    print(pd.DataFrame(records)[["scenario", "model_name", "theta_star", "initial_brier_z"]].to_string(index=False))
    print(f"Reference row {args.reference_row}: accuracy={accuracies[args.reference_row]:.4%}", flush=True)
    print(f"Prepared {len(rows)} models, {n} questions each: {output.resolve()}", flush=True)


def load_m2_bank(directory, manifest, device, require_predictor=True):
    directory = Path(directory)
    n, records = manifest["n_questions"], manifest["scenarios"]
    rows = [r["model_row"] for r in records]
    if (manifest.get("schema_version") != 1 or len(records) != manifest["n_models"]
            or len(set(rows)) != len(rows) or rows != manifest["selected_rows"]
            or set(rows) & set(manifest["excluded_rows"])
            or manifest["question_columns"] != [str(i) for i in range(n)]
            or manifest["question_columns_sha256"] != hashlib.sha256(
                json.dumps(manifest["question_columns"]).encode()).hexdigest()):
        raise ValueError("Invalid M2 model bank manifest")
    zs = {}
    for record in records:
        scenario, path = record["scenario"], directory/record["z_file"]
        if scenario != f"m2_row_{record['model_row']:04d}" or sha256(path) != record["z_sha256"]:
            raise ValueError(f"M2 identity or label checksum mismatch: {scenario}")
        frame = pd.read_csv(path)
        z = frame.to_numpy(dtype=float)
        if (z.shape != (1, n) or list(frame.columns) != manifest["question_columns"]
                or not np.isin(z, [0, 1]).all() or z.mean() != record["theta_star"]
                or z.sum() != record["n_correct"]):
            raise ValueError(f"Invalid M2 labels: {scenario}")
        zs[scenario] = torch.as_tensor(z[0], dtype=torch.float64, device=device)
    if not require_predictor:
        return manifest, zs, None, None, None
    snapshot = directory/manifest["predictor_file"]
    if sha256(snapshot) != manifest["predictor_sha256"]:
        raise ValueError("M2 predictor snapshot checksum mismatch")
    with np.load(snapshot, allow_pickle=False) as a:
        v, mu, sigma = [torch.as_tensor(a[k], dtype=torch.float64, device=device) for k in ("v", "mu", "sigma")]
    if (v.ndim != 2 or v.shape[0] != n or mu.shape != (v.shape[1],)
            or sigma.shape != (v.shape[1], v.shape[1])
            or not all(torch.isfinite(a).all() for a in (v, mu, sigma))):
        raise ValueError("Invalid M2 predictor dimensions or values")
    return manifest, zs, v, mu, sigma


def model_metadata(manifest, record):
    return dict(data_kind=M2_BANK_KIND, model_row=record["model_row"], model_name=record["model_name"],
                model_sha=record["model_sha"], reference_row=manifest["reference_row"],
                reference_theta_star=manifest["reference_theta_star"],
                accuracy_gap=record.get("accuracy_gap", manifest["accuracy_gap"]),
                n_models_selected=manifest["n_models"], question_columns_sha256=manifest["question_columns_sha256"],
                predictor_snapshot_sha256=manifest["predictor_sha256"])


# ============================================================================
# Hedged-CS and Hedged-WoR baselines
# ============================================================================

class HedgedCS:
    """Independent repeats of a running-intersection CS on a supplied grid."""

    def __init__(self, grid, n_repeats, alpha=.05, cap=.5, weight=.5):
        if not 0 < alpha < 1 or not 0 < cap < 1 or not 0 < weight < 1:
            raise ValueError("alpha, hedge cap and hedge weight must lie in (0,1)")
        if (grid.ndim != 1 or not grid.is_floating_point() or grid.numel() == 0
                or not torch.isfinite(grid).all() or (grid < 0).any() or (grid > 1).any()
                or not (grid[1:] > grid[:-1]).all() or n_repeats < 1):
            raise ValueError("Expected an increasing grid in [0,1] and positive repeat count")
        self.grid = grid.clone()
        self.population_size = None
        self.alpha, self.cap, self.weight = alpha, cap, weight
        self.t = 0
        self.log_plus = grid.new_zeros((n_repeats, len(grid)))
        self.log_minus = torch.zeros_like(self.log_plus)
        self.active = torch.ones_like(self.log_plus, dtype=torch.bool)
        self.eliminated_at = torch.zeros_like(self.log_plus, dtype=torch.int32)
        self.label_sum = grid.new_zeros(n_repeats)
        self.squared_residual_sum = grid.new_zeros(n_repeats)

    def log_capital(self):
        """Capital frozen at elimination for rejected candidates."""
        return torch.maximum(math.log(self.weight) + self.log_plus,
                             math.log1p(-self.weight) + self.log_minus)

    def update(self, labels):
        """Use only pre-observation statistics for bets, then ingest labels.

        Returns the actual pre-query statistics and number of candidates updated.
        A rejected candidate never re-enters and its two capitals stay frozen.
        """
        labels = torch.as_tensor(labels, device=self.grid.device, dtype=self.grid.dtype)
        if (labels.shape != self.label_sum.shape or not torch.isfinite(labels).all()
                or (labels < 0).any() or (labels > 1).any()):
            raise ValueError("Expected one observation in [0,1] per repeat")
        step = self.t + 1
        wor = self.population_size is not None
        if wor:
            if step > self.population_size:
                raise ValueError("Hedged-WoR cannot observe more than N labels")
            if not ((labels == 0) | (labels == 1)).all():
                raise ValueError("HedgedWoRCS uses a binary-population candidate grid")
        # At step t the denominator for sigma_hat_(t-1)^2 is t, NOT t+1.
        variance_before = (.25 + self.squared_residual_sum) / step
        mean_before = (.5 + self.label_sum) / step
        raw = torch.sqrt(2 * math.log(2 / self.alpha)
                         / (variance_before * step * math.log(step + 1)))
        counts = self.active.sum(1)
        rows, columns = self.active.nonzero(as_tuple=True)
        m = self.grid[columns]
        if wor:
            # Integer total counts avoid roundoff in N*(k/N), particularly at
            # remaining means 0/1 and at the census. Only past labels enter m_t.
            remaining = self.population_size - step + 1
            m = (self.candidate_totals[columns] - self.label_sum[rows]) / remaining
            if (m < 0).any() or (m > 1).any():
                raise AssertionError("An infeasible WoR candidate survived the previous step")
        # Division by zero gives +inf, implementing the unbounded cap at 0/1.
        plus = torch.minimum(raw[rows], self.cap / m)
        minus = torch.minimum(raw[rows], self.cap / (1 - m))
        delta = labels[rows] - m
        log_plus = torch.log1p(plus * delta)
        log_minus = torch.log1p(-minus * delta)
        if not torch.isfinite(log_plus).all() or not torch.isfinite(log_minus).all():
            raise FloatingPointError("Invalid Hedged-CS factor; no silent clipping")
        self.log_plus[rows, columns] += log_plus
        self.log_minus[rows, columns] += log_minus
        capital = torch.maximum(math.log(self.weight) + self.log_plus[rows, columns],
                                math.log1p(-self.weight) + self.log_minus[rows, columns])
        crossed = capital >= math.log(1 / self.alpha)
        rr, cc = rows[crossed], columns[crossed]
        self.active[rr, cc] = False
        self.eliminated_at[rr, cc] = step
        self.label_sum += labels
        mean_after = (.5 + self.label_sum) / (step + 1)
        # Eq. (26) centers residual X_t at mu_hat_t, after ingesting X_t.
        self.squared_residual_sum += (labels - mean_after).square()
        self.t = step
        diagnostics = dict(hedged_raw_lambda=raw, hedged_variance_before=variance_before,
                           hedged_mean_before=mean_before, candidate_evaluations=counts)
        if wor:
            # Intersect with the certainty bounds S_t <= N*m <= S_t+(N-t).
            # These exclude impossible populations and make the next m_t valid.
            # At t=N this leaves the sample mean ONLY if it is still active;
            # a previously rejected true candidate is never resurrected.
            total = self.candidate_totals[None, :]
            impossible = ((total < self.label_sum[:, None])
                          | (total > self.label_sum[:, None] + self.population_size - step))
            excluded = self.active & impossible
            self.active[excluded] = False
            self.eliminated_at[excluded] = step
            diagnostics["wor_feasibility_rejections"] = excluded.sum(1)
        return diagnostics


class HedgedWoRCS(HedgedCS):
    """Theorem 4 with recommended predictable plug-in bets, for binary labels.

    https://arxiv.org/html/2010.09686v7#S5.SS3
    Replace m by (N*m-S_(t-1))/(N-t+1) in both factors and betting caps.
    Also intersect with the deterministic finite-population feasibility interval.
    The constructor knows N and candidate totals, never the true total or labels.
    """

    def __init__(self, grid, n_repeats, population_size, alpha=.05, cap=.5, weight=.5):
        super().__init__(grid, n_repeats, alpha, cap, weight)
        if (isinstance(population_size, bool) or int(population_size) != population_size
                or population_size < 1):
            raise ValueError("population_size must be a positive integer")
        self.population_size = int(population_size)
        totals = self.grid * self.population_size
        rounded = totals.round()
        if not torch.allclose(totals, rounded, rtol=0.,
                              atol=8 * torch.finfo(grid.dtype).eps * self.population_size):
            raise ValueError("Binary WoR candidates must be integer multiples of 1/N")
        self.candidate_totals = rounded


# ============================================================================
# Asymptotic reference shapes (not numerical bounds)
# ============================================================================

def power_label(exponent):
    if np.isclose(exponent, 0):
        return "1"
    if np.isclose(exponent, .5):
        return r"t^{-1/2}"
    if np.isclose(exponent, .25):
        return r"t^{-1/4}"
    if np.isclose(exponent, 1):
        return r"t^{-1}"
    return rf"t^{{-{exponent:g}}}"


def rate_reference(t, kappa, family):
    """Leading t-order, treating the fixed-horizon logarithmic factors as constants."""
    if family == "bet":
        exponent = min(kappa, .5)
        return t ** (-exponent), power_label(exponent)
    if np.isclose(kappa, 1):
        # 1+log(t) is positive at t=1 and has the same asymptotic order as log(t).
        return (1 + np.log(t)) / t, r"(1+\log t)/t"
    exponent = min(kappa, 1)
    return t ** (-exponent), power_label(exponent)


def cumulative_reference(t, kappa):
    if np.isclose(kappa, 1):
        return 1 + np.log(t), r"1+\log t"
    if kappa > 1:
        return np.ones_like(t), "1"
    exponent = 1 - kappa
    label = "t" if np.isclose(exponent, 1) else rf"t^{{{exponent:g}}}"
    return t ** exponent, label


# ============================================================================
# Nested experiment runner, result pooling and diagnostics
# ============================================================================

KIND = "nested_fixed_z"


KEYS = ["scenario", "regime", "method"]


METHODS = {
    "ripr_uniform": ("RIPr uniform", "#332288", "o"),
    "ripr_maxmin": ("RIPr max-min", "#332288", "s"),
    "ripr_maxmax": ("RIPr max-max", "#332288", "X"),
    "bet_uniform": ("Betting uniform", "#D55E00", "o"),
    "bet_maxmin": ("Betting max-min", "#D55E00", "s"),
    "bet_maxmax": ("Betting max-max", "#D55E00", "X"),
    "hedged_uniform": ("Hedged-CS uniform", "#555555", "v"),
    "hedged_wor": ("Hedged-WoR uniform", "#AA4499", "P"),
}


DEFAULT_METHODS = tuple(m for m in METHODS if not m.endswith("_maxmax"))


QUERY_POLICIES = ("uniform", "maxmin", "maxmax")


BASELINE_METHODS = ("hedged_uniform", "hedged_wor")


METRICS = [
    "width", "covered", "anytime_covered", "hull_covered", "empty_cs",
    "active_candidates", "candidate_evaluations", "mae_z", "brier_z",
    "kl_z_uniform", "kl_z_query", "brier_z_query", "cum_kl_z_query",
    "cum_mae_z", "cum_brier_z_query", "prediction_mean", "prediction_mean_error",
    "q_min", "q_max", "query_opt_iterations", "query_opt_evaluations",
]


STOP_METRICS = ["stop_time_capped", "reached", "covered_at_eval_time",
                "covered_all_times_to_horizon", "empty_at_eval_time"]


class FAQPredictor:
    """FAQ initialization and updates adapted from skbwu/efficiently-evaluating-llms.

    Each repeat starts at the supplied historical factor mean/covariance.
    predict() precedes the current query; update() follows the evidence update.
    """

    def __init__(self, v, mu, sigma, n_repeats):
        self.v = v
        self.u = mu.repeat(n_repeats, 1)
        self.sigma = sigma.repeat(n_repeats, 1, 1)

    def predict(self, step):
        self.p = predictions(self.u, self.v)
        return self.p

    def update(self, step, indices, labels):
        self.u, self.sigma = advance_predictor(
            self.u, self.sigma, self.p, self.v, indices, labels)


class OraclePredictor:
    def __init__(self, z, n_repeats, horizon, amplitude, center, kappa):
        # "kl" here only selects a_s=a0*s^(-kappa), for BOTH CS families.
        self.sequence = OraclePredictionSequence(
            z, amplitude, kappa, horizon, mismatch_metric="kl", oracle_center=center)
        self.n_repeats = n_repeats

    def predict(self, step):
        return self.sequence(step - 1, self.n_repeats)


def load_factory(spec):
    filename, separator, name = spec.rpartition(":")
    if not separator or not Path(filename).is_file():
        raise ValueError("--predictor-factory must be /path/to/file.py:factory_name")
    module_spec = importlib.util.spec_from_file_location("nested_user_predictor", filename)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    factory = getattr(module, name)
    if not callable(factory):
        raise ValueError("Predictor factory is not callable")
    return factory


def make_predictor(args, z, v, mu, sigma, repeat_ids, scenario, kappa):
    if args.predictor == "faq":
        return FAQPredictor(v, mu, sigma, len(repeat_ids))
    if args.predictor == "oracle":
        return OraclePredictor(z, len(repeat_ids), args.max_steps,
                               args.oracle_amplitude, args.oracle_center, kappa)
    context = dict(v=v.clone(), mu=mu.clone(), sigma=sigma.clone(),
                   n_questions=z.numel(), n_repeats=len(repeat_ids),
                   repeat_ids=repeat_ids.copy(), device=z.device,
                   seed=args.seed, scenario=scenario, max_steps=args.max_steps,
                   parameters=json.loads(args.predictor_kwargs),
                   z=z.clone() if args.custom_oracle else None)
    predictor = load_factory(args.predictor_factory)(context)
    if not callable(getattr(predictor, "predict", None)):
        raise ValueError("Custom factory must return an object with predict(step)")
    return predictor


def load_data(directory, device, require_predictor=True):
    """Read both existing multi-scenario and single-z oracle manifests."""
    directory = Path(directory)
    manifest = read_data_manifest(directory / "manifest.json")
    if manifest.get("benchmark_kind") == M2_BANK_KIND:
        return load_m2_bank(directory, manifest, device, require_predictor)
    if not require_predictor:
        # Baseline-only runs need the verified labels, not a factor snapshot.
        n = manifest["n_questions"]
        records = manifest.get("scenarios", [dict(manifest, scenario="fixed_z")])
        zs = {}
        for record in records:
            path = directory / record["z_file"]
            if sha256(path) != record["z_sha256"]:
                raise ValueError(f"Fixed z checksum mismatch: {path}")
            values = pd.read_csv(path).to_numpy(dtype=float)
            if (values.shape != (1, n) or not np.isin(values, [0, 1]).all()
                    or values.mean() != record["theta_star"] or record["scenario"] in zs):
                raise ValueError(f"Invalid fixed labels or duplicate scenario: {path}")
            zs[record["scenario"]] = torch.as_tensor(values[0], dtype=torch.float64, device=device)
        return manifest, zs, None, None, None
    if "scenarios" in manifest:
        return load_prepared(directory, device)
    if manifest.get("benchmark_kind") not in (BENCHMARK_KIND, LEGACY_BENCHMARK_KIND):
        raise ValueError("Use an existing prepared fixed-z or ideal fixed-z dataset")
    for prefix in ("predictor", "z"):
        if sha256(directory / manifest[f"{prefix}_file"]) != manifest[f"{prefix}_sha256"]:
            raise ValueError(f"{prefix} checksum mismatch")
    with np.load(directory / manifest["predictor_file"], allow_pickle=False) as a:
        v, mu, sigma = [torch.as_tensor(a[k], dtype=torch.float64, device=device)
                        for k in ("v", "mu", "sigma")]
    values = pd.read_csv(directory / manifest["z_file"]).to_numpy(dtype=float)
    if (values.shape != (1, len(v)) or not np.isin(values, [0, 1]).all()
            or mu.shape != (v.shape[1],) or sigma.shape != (v.shape[1], v.shape[1])
            or not all(torch.isfinite(x).all() for x in (v, mu, sigma))):
        raise ValueError("Invalid z / predictor dimensions or values")
    z = torch.as_tensor(values[0], dtype=torch.float64, device=device)
    return manifest, {"fixed_z": z}, v, mu, sigma


def scenario_stream(name):
    name = canonical_scenario(name)
    if name in SCENARIOS:
        return list(SCENARIOS).index(name)
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def active_log_increments(p, q, indices, labels, grid, h, active, family, batch_size):
    """Packed ragged candidates: no padding and no previously rejected m solved.

    Each solver batch has at most batch_size (repeat, m) pairs, so its temporary
    tensors are O(batch_size * N), independent of the total repeat count.
    """
    pairs = active.nonzero(as_tuple=False)
    increments = torch.empty(len(pairs), dtype=p.dtype, device=p.device)
    for start in range(0, len(pairs), batch_size):
        stop = min(start + batch_size, len(pairs))
        rows, columns = pairs[start:stop].unbind(1)
        pp, qq = p[rows], q[rows]
        targets = grid[columns, None]
        ii, yy = indices[rows], labels[rows]
        sampled_p = pp.gather(1, ii)
        if family == "ripr":
            # Handle null means 0/1 exactly; impossible labels reject at once.
            boundary = (targets[:, 0] == 0) | (targets[:, 0] == 1)
            denominator = targets.clone()
            interior = ~boundary
            if interior.any():
                dual = solve_ripr_dual_lambda_batch(
                    pp[interior], qq[interior], targets[interior])
                projected = compute_ripr_point_given_dual_lambda(
                    pp[interior], qq[interior], dual)
                if projected.ndim == 3:
                    projected = projected[:, 0, :]
                denominator[interior] = projected.gather(1, ii[interior])
            # torch.where avoids 0*log(0) at exact boundary projections.
            log_num = torch.where(yy.bool(), sampled_p.log(), torch.log1p(-sampled_p))
            log_den = torch.where(yy.bool(), denominator.log(), torch.log1p(-denominator))
            inc = (log_num - log_den)[:, 0]
        else:
            lam = solve_optimal_betting_lambda_batch(pp, qq, targets, h[rows])
            phi = pp.mean(1, keepdim=True) + (yy - sampled_p) / (p.shape[1] * qq.gather(1, ii))
            factor = 1 + lam * (phi - targets)
            if not torch.isfinite(factor).all() or (factor <= 0).any():
                raise FloatingPointError("Invalid betting e-factor; no silent clipping")
            inc = factor.log()[:, 0]
        if torch.isnan(inc).any() or torch.isneginf(inc).any():
            raise FloatingPointError("Invalid log e-value")
        increments[start:stop] = inc
    return pairs, increments


def update_nested(log_wealth, active, eliminated_at, pairs, increments, step, alpha):
    rows, columns = pairs.unbind(1)
    log_wealth[rows, columns] += increments
    crossed = log_wealth[rows, columns] >= math.log(1 / alpha)
    rr, cc = rows[crossed], columns[crossed]
    active[rr, cc] = False
    eliminated_at[rr, cc] = step


def nested_endpoints(active, grid):
    nonempty = active.any(1)
    lower = torch.where(active, grid[None, :], torch.inf).min(1).values
    upper = torch.where(active, grid[None, :], -torch.inf).max(1).values
    # Empty-set diameter is zero, but it is NEVER counted as a precision hit.
    width = torch.where(nonempty, upper - lower, 0.)
    return lower, upper, width, nonempty


def maxmax_distribution(predictions, lower, upper, family, uniform_weight=0.05,
                        opt_steps=25, learning_rate=0.5, *, initial_q=None,
                        opt_rtol=OPT_RTOL, opt_atol=OPT_ATOL):
    """Numerically maximize max(G(q,L), G(q,U)) on the floored simplex.

    Optimize each endpoint independently. Passing identical endpoints to the
    existing max-min solver makes its objective exactly the single-endpoint
    growth, while retaining its multistart search and actual-growth line search.
    Re-score both returned distributions on BOTH endpoints, then choose the
    larger max-growth. No global optimality is claimed for the numerical search.
    Only predictions and pre-query CS endpoints enter this routine.
    """
    p = predictions.detach().to(dtype=torch.float64)
    if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < 1:
        raise ValueError("predictions must have shape (repeats, questions).")
    lower = torch.as_tensor(lower, dtype=p.dtype, device=p.device)
    upper = torch.as_tensor(upper, dtype=p.dtype, device=p.device)
    repeats = p.shape[0]
    if lower.shape != (repeats,) or upper.shape != lower.shape:
        raise ValueError("CS endpoints must have one entry per repeat.")
    if (not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
            or (lower < 0).any() or (upper > 1).any() or (lower > upper).any()):
        raise ValueError("Max-max requires nonempty CS endpoints in [0,1].")
    if family not in ("ripr", "bet"):
        raise ValueError(f"Unknown CS family: {family}")
    warm = None
    if initial_q is not None:
        initial_q = torch.as_tensor(initial_q, dtype=p.dtype, device=p.device)
        if initial_q.shape != p.shape:
            raise ValueError("initial_q must match predictions.shape")
        warm = initial_q.repeat(2, 1)
    pp = p.repeat(2, 1)
    targets = torch.cat((lower, upper))
    qs, hs, info = maxmin_distribution(
        pp, targets, targets, family, uniform_weight, opt_steps, learning_rate,
        initial_q=warm, opt_rtol=opt_rtol, opt_atol=opt_atol)
    # A finite-iteration branch may incidentally be better on the other endpoint;
    # selecting by the actual max objective also covers that case.
    endpoints = torch.stack((lower, upper), dim=1).repeat(2, 1)
    with torch.no_grad():
        if family == "ripr":
            _, growths, _, _ = compute_ripr_endpoint_stats(pp, qs, endpoints)
        else:
            lam = solve_optimal_betting_lambda_batch(pp, qs, endpoints, hs)
            growths = compute_betting_expected_log_growths_batch(pp, qs, endpoints, lam)
    if not torch.isfinite(growths).all():
        raise FloatingPointError("Nonfinite max-max endpoint growth")
    values = growths.amax(1).reshape(2, repeats)
    branch = values.argmax(0)
    rows = branch * repeats + torch.arange(repeats, device=p.device)
    selected = growths[rows]
    uniform_value = info["uniform_candidate_growth"].reshape(2, repeats).amax(0)
    return qs[rows].detach(), hs[rows], {
        "growth_lower": selected[:, 0], "growth_upper": selected[:, 1],
        "min_growth": selected.amin(1), "max_growth": selected.amax(1),
        "selected_endpoint": selected.argmax(1), "optimizer_branch": branch,
        "uniform_candidate_growth": uniform_value,
        "initial_candidate_growth": info["initial_candidate_growth"].reshape(2, repeats).amax(0),
        "gain_over_uniform": selected.amax(1) - uniform_value,
        "optimizer_iterations": info["optimizer_iterations"].reshape(2, repeats).sum(0),
        "optimizer_evaluations": info["optimizer_evaluations"].reshape(2, repeats).sum(0) + 2,
        "optimizer_status": info["optimizer_status"][rows],
        "optimizer_start": info["optimizer_start"][rows],
        "optimizer_budget_hit": info["optimizer_budget_hit"].reshape(2, repeats).any(0),
    }


def simulate(z, predictor, family, policy, uniforms, args, progress_label=""):
    if family in ("hedged", "hedged_wor"):
        if policy != "uniform":
            raise ValueError("The no-side-information Hedged-CS baseline requires uniform sampling")
        return simulate_hedged(z, uniforms, args, progress_label,
                               without_replacement=family == "hedged_wor")
    if policy not in QUERY_POLICIES:
        raise ValueError(f"Unknown querying policy: {policy}")
    repeats, horizon = uniforms.shape
    n, theta = z.numel(), float(z.mean())
    true_index = int(z.sum())
    grid = torch.arange(n + 1, dtype=z.dtype, device=z.device) / n
    active = torch.ones((repeats, n + 1), dtype=torch.bool, device=z.device)
    wealth = torch.zeros((repeats, n + 1), dtype=z.dtype, device=z.device)
    eliminated = torch.zeros_like(wealth, dtype=torch.int32)
    lower, upper = torch.zeros(repeats, device=z.device), torch.ones(repeats, device=z.device)
    previous_q = None
    cumulative = torch.zeros((3, repeats), dtype=z.dtype, device=z.device)
    history = {key: np.empty((horizon, repeats)) for key in METRICS + [
        "lower", "upper", "queried_index", "queried_label", "sampled_q", "h",
        "log_wealth_true_stopped",
    ]}
    for t in range(1, horizon + 1):
        # Clone: a custom predictor may mutate its internal state in update().
        with torch.no_grad():
            p = torch.as_tensor(predictor.predict(t), dtype=z.dtype, device=z.device).detach().clone()
        if p.shape == (n,):
            p = p[None, :].expand(repeats, -1)
        if (p.shape != (repeats, n) or not torch.isfinite(p).all()
                or (p < PROB_EPS).any() or (p > 1 - PROB_EPS).any()):
            raise ValueError(f"predict({t}) must be (N,) or (repeats,N), in [1e-6,1-1e-6]")
        nonempty = active.any(1)
        q = torch.full_like(p, 1 / n)
        h = torch.full((repeats,), 1 / n, dtype=z.dtype, device=z.device)
        iterations, evaluations = torch.zeros_like(h), torch.zeros_like(h)
        if policy in ("maxmin", "maxmax") and nonempty.any():
            optimizer = maxmax_distribution if policy == "maxmax" else maxmin_distribution
            q_live, h_live, info = optimizer(
                p[nonempty], lower[nonempty], upper[nonempty], family,
                args.uniform_weight, args.grow_opt_steps, args.grow_opt_lr,
                initial_q=previous_q[nonempty] if previous_q is not None else None,
                opt_rtol=args.grow_opt_rtol, opt_atol=args.grow_opt_atol)
            q[nonempty], h[nonempty] = q_live, h_live
            if info:
                iterations[nonempty] = info["optimizer_iterations"].to(iterations.dtype)
                evaluations[nonempty] = info["optimizer_evaluations"].to(evaluations.dtype)
        previous_q = q.detach()
        if (q < h[:, None] - 1e-12).any() or not torch.allclose(q.sum(1), torch.ones_like(h)):
            raise AssertionError("Invalid query probabilities")
        # Current response is drawn only AFTER the predictor and q are fixed.
        indices = sample_from_uniforms(q, torch.as_tensor(
            uniforms[:, t-1:t].copy(), dtype=z.dtype, device=z.device))
        labels = z[indices]
        count_before = active.sum(1)
        with torch.no_grad():
            pairs, inc = active_log_increments(p, q, indices, labels, grid, h, active,
                                               family, args.candidate_batch_size)
            update_nested(wealth, active, eliminated, pairs, inc, t, args.alpha)
        lower, upper, width, nonempty = nested_endpoints(active, grid)
        covered = active[:, true_index]  # exact candidate membership, not interval hull
        error = p - z[None, :]
        loss = torch.where(z[None, :].bool(), -p.log(), -torch.log1p(-p))
        kl, mae, brier = (q * loss).sum(1), error.abs().mean(1), (q * error.square()).sum(1)
        cumulative += torch.stack((kl, mae, brier))
        values = dict(width=width, lower=lower, upper=upper, covered=covered,
                      anytime_covered=covered, hull_covered=nonempty & (lower <= theta) & (theta <= upper),
                      empty_cs=~nonempty, active_candidates=active.sum(1), candidate_evaluations=count_before,
                      mae_z=mae, brier_z=error.square().mean(1), kl_z_uniform=loss.mean(1),
                      kl_z_query=kl, brier_z_query=brier, cum_kl_z_query=cumulative[0],
                      cum_mae_z=cumulative[1], cum_brier_z_query=cumulative[2],
                      prediction_mean=p.mean(1), prediction_mean_error=p.mean(1)-theta,
                      q_min=q.min(1).values, q_max=q.max(1).values, h=h,
                      query_opt_iterations=iterations, query_opt_evaluations=evaluations,
                      queried_index=indices[:, 0], queried_label=labels[:, 0],
                      sampled_q=q.gather(1, indices)[:, 0], log_wealth_true_stopped=wealth[:, true_index])
        for key, value in values.items():
            history[key][t-1] = value.detach().cpu().numpy()
        if callable(getattr(predictor, "update", None)):
            with torch.no_grad():
                predictor.update(t, indices.clone(), labels.clone())
        if progress_label and (t % args.progress_every == 0 or t == horizon):
            print(f"{progress_label}: {t}/{horizon}, mean active={float(active.sum(1).double().mean()):.1f}/{n+1}", flush=True)
    return history, eliminated.cpu().numpy()


def simulate_hedged(z, uniforms, args, progress_label="", *, without_replacement=False):
    """Uniform WR/WoR; only sampled labels enter the observation-only CS state."""
    repeats, horizon = uniforms.shape
    n, true_index = z.numel(), int(z.sum())
    if not np.isfinite(uniforms).all() or (uniforms < 0).any() or (uniforms >= 1).any():
        raise ValueError("Uniform draws must lie in [0,1)")
    grid = torch.arange(n + 1, dtype=z.dtype, device=z.device) / n
    if without_replacement:
        if horizon > n:
            raise ValueError("Hedged-WoR requires max_steps <= N (sampling without replacement)")
        state = HedgedWoRCS(grid, repeats, n, args.alpha, args.hedge_cap, args.hedge_weight)
        # Partial Fisher-Yates: choose a uniform remaining slot, swap with the
        # last remaining slot, and shrink the pool. O(1) selection per repeat.
        pool = torch.arange(n, device=z.device).repeat(repeats, 1)
        row_ids = torch.arange(repeats, device=z.device)
    else:
        state = HedgedCS(grid, repeats, args.alpha, args.hedge_cap, args.hedge_weight)
        # Cache exactly the CDF used by sample_from_uniforms: paired
        # WR draws match other uniform methods, including boundary conventions.
        cdf = torch.full((repeats, n), 1 / n, dtype=z.dtype, device=z.device).cumsum(1)
        cdf[:, -1] = 1.
    extra = ["lower", "upper", "queried_index", "queried_label", "sampled_q", "h",
             "log_wealth_true_stopped", "log_wealth_plus_true_stopped", "log_wealth_minus_true_stopped",
             "hedged_raw_lambda", "hedged_variance_before", "hedged_mean_before"]
    if without_replacement:
        extra += ["wor_remaining_questions_before", "wor_remaining_mean_true_before",
                  "wor_feasibility_rejections"]
    # Prediction mismatch is inapplicable, not zero, for this method.
    history = {key: np.full((horizon, repeats), np.nan) for key in METRICS + extra}
    probability = z.new_full((repeats,), 1 / n)
    zero = z.new_zeros(repeats)
    for t in range(1, horizon + 1):
        draws = torch.as_tensor(uniforms[:, t-1:t].copy(), dtype=z.dtype, device=z.device)
        if without_replacement:
            remaining = n - t + 1
            slots = (draws[:, 0] * remaining).floor().long()
            indices = pool[row_ids, slots].clone()
            pool[row_ids, slots] = pool[:, remaining-1].clone()
            probability = z.new_full((repeats,), 1 / remaining)
            # Evaluation diagnostic only: not passed into the CS state or sampler.
            remaining_mean_true = (true_index - state.label_sum) / remaining
        else:
            indices = torch.searchsorted(cdf, draws.contiguous(), right=True)[:, 0]
        labels = z[indices]
        with torch.no_grad():
            diagnostics = state.update(labels)
        lower, upper, width, nonempty = nested_endpoints(state.active, grid)
        covered = state.active[:, true_index]
        minimum_q = zero if without_replacement and t > 1 else probability
        values = dict(diagnostics, width=width, lower=lower, upper=upper, covered=covered,
                      anytime_covered=covered,
                      hull_covered=nonempty & (lower <= grid[true_index]) & (grid[true_index] <= upper),
                      empty_cs=~nonempty,
                      active_candidates=state.active.sum(1), q_min=minimum_q, q_max=probability,
                      h=minimum_q, sampled_q=probability, queried_index=indices, queried_label=labels,
                      query_opt_iterations=zero, query_opt_evaluations=zero,
                      log_wealth_plus_true_stopped=state.log_plus[:, true_index],
                      log_wealth_minus_true_stopped=state.log_minus[:, true_index],
                      log_wealth_true_stopped=torch.maximum(
                          math.log(args.hedge_weight) + state.log_plus[:, true_index],
                          math.log1p(-args.hedge_weight) + state.log_minus[:, true_index]))
        if without_replacement:
            values.update(wor_remaining_questions_before=z.new_full((repeats,), remaining),
                          wor_remaining_mean_true_before=remaining_mean_true)
        for key, value in values.items():
            history[key][t-1] = value.detach().cpu().numpy()
        if progress_label and (t % args.progress_every == 0 or t == horizon):
            print(f"{progress_label}: {t}/{horizon}, mean active={float(state.active.sum(1).double().mean()):.1f}/{n+1}", flush=True)
    return history, state.eliminated_at.cpu().numpy()


def make_frames(history, repeat_ids, epsilons, metadata):
    horizon, repeats = history["width"].shape
    steps = pd.DataFrame({k: v.reshape(-1) for k, v in history.items()})
    steps["step"] = np.repeat(np.arange(1, horizon + 1), repeats)
    steps["repeat_id"] = np.tile(repeat_ids, horizon)
    for key in ("queried_index", "queried_label", "active_candidates", "candidate_evaluations",
                "query_opt_iterations", "query_opt_evaluations"):
        if key in steps:
            steps[key] = steps[key].astype(np.int64)
    stops = []
    for epsilon in epsilons:
        hit = (history["width"] <= epsilon) & ~history["empty_cs"].astype(bool)
        reached = hit.any(0)
        indices = np.where(reached, hit.argmax(0), horizon - 1)
        for j, rid in enumerate(repeat_ids):
            stops.append(dict(epsilon=epsilon, repeat_id=rid, stop_time_capped=indices[j]+1,
                              reached=reached[j], covered_at_eval_time=history["covered"][indices[j], j],
                              covered_all_times_to_horizon=history["anytime_covered"][-1, j],
                              empty_at_eval_time=history["empty_cs"][indices[j], j]))
    stops = pd.DataFrame(stops)
    for frame in (steps, stops):
        for key, value in metadata.items():
            frame[key] = value
    return steps, stops


def summarize(frame, group_keys, metrics):
    grouped = frame.groupby(group_keys, sort=True, dropna=False)
    result = grouped.size().rename("n_repeats").to_frame()
    for metric in metrics:
        result[f"{metric}_mean"] = grouped[metric].mean()
        result[f"{metric}_se"] = grouped[metric].std(ddof=1) / np.sqrt(result.n_repeats)
    return result.reset_index()


def pool_summaries(frames, group_keys, metrics):
    """Pool exact first/second moments; never average SEs or batch means equally."""
    frame = pd.concat(frames, ignore_index=True)
    frame["n_repeats"] = frame.n_repeats.astype(int)
    for metric in metrics:
        n, mu, se = frame.n_repeats, frame[f"{metric}_mean"], frame[f"{metric}_se"]
        frame[f"{metric}_sum"] = n * mu
        frame[f"{metric}_ss"] = np.where(n > 1, se.fillna(0)**2*n*(n-1), 0) + n*mu**2
    cols = ["n_repeats"] + [f"{m}_{suffix}" for m in metrics for suffix in ("sum", "ss")]
    result = frame.groupby(group_keys, dropna=False, sort=True)[cols].sum(min_count=1)
    n = result.n_repeats
    for metric in metrics:
        mu = result.pop(f"{metric}_sum") / n
        ss = result.pop(f"{metric}_ss")
        result[f"{metric}_mean"] = mu
        result[f"{metric}_se"] = np.sqrt(((ss-n*mu**2).clip(lower=0)/(n-1).replace(0, np.nan))/n)
    return result.reset_index()


def add_metadata(frame, metadata):
    for key, value in metadata.items():
        if key not in frame:
            frame[key] = value
    return frame


def run(args):
    torch.set_num_threads(args.threads)
    needs_predictor = any(m not in BASELINE_METHODS for m in args.methods)
    validate_uniform_weight(args.uniform_weight)
    if not 0 < args.alpha < 1 or not all(0 < e < 1 for e in args.epsilons):
        raise ValueError("alpha and epsilons must lie in (0,1)")
    if (not math.isfinite(args.grow_opt_lr) or args.grow_opt_lr <= 0
            or any(not math.isfinite(v) or v < 0 for v in (args.grow_opt_rtol, args.grow_opt_atol))):
        raise ValueError("Invalid optimizer tolerances / learning rate")
    if not 0 < args.hedge_cap < 1 or not 0 < args.hedge_weight < 1:
        raise ValueError("--hedge-cap and --hedge-weight must lie in (0,1)")
    if needs_predictor and args.predictor == "custom":
        if not args.predictor_factory or not isinstance(json.loads(args.predictor_kwargs), dict):
            raise ValueError("Custom predictor requires --predictor-factory and JSON object kwargs")
        load_factory(args.predictor_factory)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    manifest, zs, v, mu, sigma = load_data(args.data_dir, device, require_predictor=needs_predictor)
    n = next(iter(zs.values())).numel()
    scenarios = [canonical_scenario(s) for s in args.scenarios] if args.scenarios else list(zs)
    if len(set(scenarios)) != len(scenarios) or set(scenarios) - set(zs):
        raise ValueError(f"Choose distinct scenarios from {list(zs)}")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Duplicate methods")
    args.max_steps = args.max_steps or n
    if "hedged_wor" in args.methods and args.max_steps > n:
        raise ValueError("Hedged-WoR requires --max-steps <= N; select WR methods for longer runs")
    args.epsilons = sorted(set(args.epsilons))
    if needs_predictor and args.uniform_weight / n < RIPR_Q_EPS:
        raise ValueError("Query floor below solver precision")
    kappas = sorted(set(args.kappas)) if needs_predictor and args.predictor == "oracle" else [None]
    if needs_predictor and args.predictor == "oracle":
        for scenario in scenarios:
            for kappa in kappas:
                OraclePredictor(zs[scenario], 1, args.max_steps, args.oracle_amplitude, args.oracle_center, kappa)
    output = empty_output_directory(args.output_dir)
    config = vars(args).copy()
    config.update(experiment_kind=KIND, schema_version=3, status="running", n_questions=n,
                  scenarios=scenarios, data_dir=str(Path(args.data_dir).resolve()),
                  manifest_sha256=sha256(Path(args.data_dir)/"manifest.json"), benchmark=manifest,
                  cs_definition="intersection_{s<=t}{m:W_s(m)<1/alpha}",
                  coverage_definition="exact surviving theta_star candidate; equals through-time coverage",
                  width_definition="max(active)-min(active); empty diameter=0 but empty is not a precision hit",
                  stopping_definition="first nonempty nested CS with full width <= epsilon, capped at max_steps",
                  betting_range="full valid range (same as existing implementation)",
                  querying="uniform, endpoint max-min, or endpoint max-max; nested CS endpoints; uniform after empty CS",
                  maxmax_definition="max_q max(G(q,L),G(q,U)); independent endpoint multistart searches, then re-score both candidates on both endpoints; same eta/N floor and betting range as max-min",
                  sampling={m: "uniform_without_replacement" if m == "hedged_wor"
                            else "with_replacement" for m in args.methods},
                  random_streams="SeedSequence([seed, stable scenario index, repeat_id]); paired across methods/kappa",
                  prediction_timing="predict before query; update only after current queried label",
                  hedged_definition="Section 4.4 eqs (24)-(26); max(weight*K_plus,(1-weight)*K_minus); nested",
                  hedged_predictor="none; pre-observation label statistics only; one run per scenario, independent of kappa",
                  hedged_wor_definition="Theorem 4; m_t=(N*m-S_(t-1))/(N-t+1); intersect nested CS with [S_t/N,(S_t+N-t)/N]",
                  hedged_wor_sampling="uniform among remaining indices via partial Fisher-Yates; shared uniforms but different indices from WR",
                  device=str(device), source_sha256={Path(__file__).name: sha256(Path(__file__))})
    if needs_predictor and args.predictor == "custom":
        config["predictor_source_sha256"] = sha256(args.predictor_factory.rpartition(":")[0])
    write_json(output/"config.json", config)
    case_steps, case_stops = [], []
    all_ids = np.arange(args.repeat_start, args.repeat_start + args.n_repeats)
    for scenario in scenarios:
        z = zs[scenario]
        record = next((r for r in manifest.get("scenarios", []) if r["scenario"] == scenario), manifest)
        for kappa in kappas:
            for method in args.methods:
                baseline = method in BASELINE_METHODS
                if baseline and kappa != kappas[0]:
                    continue  # Same data/stream: baseline is independent of oracle kappa.
                family, policy = (("hedged_wor" if method == "hedged_wor" else "hedged", "uniform")
                                  if baseline else method.split("_", 1))
                regime = ("no_side_information" if baseline else args.predictor if kappa is None
                          else f"oracle_k{kappa:g}".replace(".", "p"))
                oracle_case = kappa is not None and not baseline
                metadata = dict(scenario=scenario, regime=regime, method=method,
                                predictor="none" if baseline else args.predictor,
                                kappa=kappa if oracle_case else np.nan,
                                oracle_amplitude=args.oracle_amplitude if oracle_case else np.nan,
                                oracle_center=args.oracle_center if oracle_case else np.nan,
                                hedge_cap=args.hedge_cap if baseline else np.nan,
                                hedge_weight=args.hedge_weight if baseline else np.nan,
                                sampling="without_replacement" if method == "hedged_wor" else "with_replacement",
                                theta_star=float(z.mean()), z_sha256=record["z_sha256"],
                                n_questions=n, horizon=args.max_steps, alpha=args.alpha)
                if manifest.get("benchmark_kind") == M2_BANK_KIND:
                    metadata.update(model_metadata(manifest, record))
                stem = f"{scenario}_{regime}_{method}"
                batch_steps, batch_stops, timing = [], [], []
                print(f"Running {stem}: N={n}, repeats={len(all_ids)}, H={args.max_steps}", flush=True)
                for start in range(0, len(all_ids), args.repeat_batch_size):
                    ids = all_ids[start:start+args.repeat_batch_size]
                    predictor = None if baseline else make_predictor(args, z, v, mu, sigma, ids, scenario, kappa)
                    uniforms = query_uniforms(args.seed, scenario_stream(scenario), ids, args.max_steps)
                    begin = time.perf_counter()
                    history, eliminated = simulate(z, predictor, family, policy, uniforms, args,
                                                   f"{stem} repeats {ids[0]}--{ids[-1]}")
                    elapsed = time.perf_counter() - begin
                    steps, stops = make_frames(history, ids, args.epsilons, metadata)
                    for suffix, frame in (("steps", steps), ("stops", stops)):
                        frame.to_csv(output/f"{stem}_{suffix}.csv", mode="w" if start == 0 else "a",
                                     header=start == 0, index=False)
                    np.savez_compressed(output/f"{stem}_eliminations_{ids[0]}_{ids[-1]}.npz",
                                        repeat_ids=ids, eliminated_at=eliminated)
                    batch_steps.append(summarize(steps, KEYS+["step"], METRICS))
                    batch_stops.append(summarize(stops, KEYS+["epsilon"], STOP_METRICS))
                    timing.append(dict(repeat_start=int(ids[0]), n_repeats=len(ids), wall_seconds=elapsed,
                                       candidate_evaluations=int(history["candidate_evaluations"].sum()),
                                       dense_candidate_evaluations=len(ids)*args.max_steps*(n+1)))
                ss = add_metadata(pool_summaries(batch_steps, KEYS+["step"], METRICS), metadata)
                ts = add_metadata(pool_summaries(batch_stops, KEYS+["epsilon"], STOP_METRICS), metadata)
                case_steps.append(ss)
                case_stops.append(ts)
                pd.DataFrame(timing).to_csv(output/f"{stem}_timing.csv", index=False)
                # Persist summaries after each completed method; incomplete runs are not plotted.
                pd.concat(case_steps).to_csv(output/"step_summary.csv", index=False)
                pd.concat(case_stops).to_csv(output/"stopping_summary.csv", index=False)
    config["status"] = "complete"
    write_json(output/"config.json", config)
    if not args.no_plots:
        plot_results(output, output/"plots", args.anchor_step)
    print(f"Results: {output.resolve()}", flush=True)


def load_results(root):
    """Combine disjoint completed run parts; reject overlapping Monte Carlo trials."""
    roots = [Path(p) for p in root] if isinstance(root, (list, tuple)) else [Path(root)]
    configs = []
    for directory in roots:
        configs.extend([directory/"config.json"] if (directory/"config.json").exists()
                       else sorted(directory.rglob("config.json")))
    steps, stops, sources, seen, signatures = [], [], [], set(), {}
    for path in configs:
        cfg = json.loads(path.read_text())
        if cfg.get("experiment_kind") != KIND:
            continue
        if cfg.get("status") != "complete":
            raise ValueError(f"Incomplete run: {path}")
        ss, ts = (pd.read_csv(path.parent/name) for name in ("step_summary.csv", "stopping_summary.csv"))
        for frame in (ss, ts):
            frame["scenario"] = frame.scenario.map(canonical_scenario)
        if set(map(tuple, ss[KEYS].drop_duplicates().to_numpy())) != set(map(tuple, ts[KEYS].drop_duplicates().to_numpy())):
            raise ValueError(f"Step/stopping cases disagree: {path}")
        for case, group in ss.groupby(KEYS):
            signature = json.dumps({k: cfg.get(k) for k in (
                "manifest_sha256", "alpha", "max_steps", "uniform_weight", "grow_opt_steps", "grow_opt_lr",
                "grow_opt_rtol", "grow_opt_atol", "predictor", "oracle_amplitude", "oracle_center",
                "predictor_source_sha256", "predictor_kwargs", "custom_oracle", "betting_range", "source_sha256", "epsilons",
                "hedge_cap", "hedge_weight",
            )}, sort_keys=True)
            if case in signatures and signatures[case] != signature:
                raise ValueError(f"Incompatible experiment settings for {case}")
            signatures[case] = signature
            expected_steps = np.arange(1, cfg["max_steps"]+1)
            if (not np.array_equal(group.sort_values("step").step, expected_steps)
                    or not group.n_repeats.eq(cfg["n_repeats"]).all()):
                raise ValueError(f"Incomplete summary for {case} in {path}")
            stopping_group = ts.loc[(ts[KEYS] == pd.Series(case, index=KEYS)).all(axis=1)].sort_values("epsilon")
            if (len(stopping_group) != len(cfg["epsilons"])
                    or not np.allclose(stopping_group.epsilon.to_numpy(), cfg["epsilons"])
                    or not stopping_group.n_repeats.eq(cfg["n_repeats"]).all()):
                raise ValueError(f"Incomplete stopping summary for {case} in {path}")
            for rid in range(cfg["repeat_start"], cfg["repeat_start"]+cfg["n_repeats"]):
                trial = (*case, cfg["seed"], rid)
                if trial in seen:
                    raise ValueError(f"Duplicate repeat {trial}; do not pool reruns of the same seeds")
                seen.add(trial)
        steps.append(ss)
        stops.append(ts)
        sources.append(str(path.resolve()))
    if not steps:
        raise FileNotFoundError(f"No completed nested experiment in {root}")
    metadata_cols = ["predictor", "kappa", "oracle_amplitude", "oracle_center", "theta_star",
                     "z_sha256", "n_questions", "horizon", "alpha", "hedge_cap", "hedge_weight", "sampling"]
    for column in M2_METADATA:
        if any(column in frame for frame in steps):
            metadata_cols.append(column)
            for frame in steps:
                if column not in frame:
                    frame[column] = np.nan
    for frame in steps:
        for column in ("hedge_cap", "hedge_weight"):
            if column not in frame:
                frame[column] = np.nan
        if "sampling" not in frame:
            frame["sampling"] = np.where(frame.method == "hedged_wor", "without_replacement", "with_replacement")
    metadata = pd.concat(steps)[KEYS+metadata_cols].drop_duplicates()
    if metadata.duplicated(KEYS).any():
        raise ValueError("Conflicting scenario metadata")
    for _, group in metadata.groupby(["scenario", "regime"]):
        for key in metadata_cols:
            if key == "sampling":
                continue  # WR and WoR are intentionally different baselines.
            if group[key].nunique(dropna=False) != 1:
                raise ValueError(f"Methods use different {key}; plot matched experiments together")
    ss = pool_summaries(steps, KEYS+["step"], METRICS).merge(metadata, on=KEYS, validate="many_to_one")
    ts = pool_summaries(stops, KEYS+["epsilon"], STOP_METRICS).merge(metadata, on=KEYS, validate="many_to_one")
    return ss, ts, sources


def export_results(args):
    """Export completed simulations in the compact format shipped under data/."""
    step, stop, sources = load_results(args.results_dir)
    columns = ["scenario", "regime", "method", "step", "n_repeats", "predictor",
               "kappa", "oracle_amplitude", "oracle_center", "theta_star",
               "z_sha256", "n_questions", "horizon", "alpha", "sampling"]
    metrics = ["width", "anytime_covered", "empty_cs", "mae_z", "kl_z_query"]
    if args.experiment == "m2_comparison":
        metrics += ["cum_mae_z", "cum_kl_z_query", "brier_z"]
        columns += [column for column in M2_METADATA if column in step]
    columns += [f"{metric}_{stat}" for metric in metrics for stat in ("mean", "se")]
    root = Path(args.output_dir)
    target = empty_output_directory(root / args.experiment)
    step[columns].to_csv(target / "step_summary.csv.gz", index=False,
                         compression={"method": "gzip", "mtime": 0})
    stop.to_csv(target / "stopping_summary.csv.gz", index=False,
                compression={"method": "gzip", "mtime": 0})
    records = []
    source_roots = ([Path(p).resolve() for p in args.results_dir]
                    if isinstance(args.results_dir, (list, tuple))
                    else [Path(args.results_dir).resolve()])
    for source in sources:
        config = json.loads(Path(source).read_text())
        config.pop("benchmark", None)
        config["exported_scenarios"] = [canonical_scenario(s) for s in config["scenarios"]]
        source_root = next(p for p in source_roots if Path(source).is_relative_to(p))
        relative = Path(source).relative_to(source_root)
        if len(source_roots) > 1:
            relative = Path(source_root.name) / relative
        config["original_config"] = str(relative)
        config["original_config_sha256"] = sha256(source)
        # Absolute runtime paths are not needed to reconstruct the plotted values.
        config.pop("data_dir", None)
        config.pop("output_dir", None)
        records.append(config)
    write_json(target / "source_runs.json", records)
    checksums = {str(path.relative_to(root)): sha256(path)
                 for path in sorted(root.rglob("*"))
                 if path.is_file() and path != root / "checksums.json"
                 and not any(part.startswith(".") for part in path.relative_to(root).parts)}
    write_json(root / "checksums.json", checksums)
    print(f"Exported {len(step)} step rows and {len(stop)} stopping rows to {target.resolve()}")


def coverage_intervals(step):
    from scipy.stats import beta
    step = step.copy()
    n, p = step.n_repeats.to_numpy(int), step.anytime_covered_mean.to_numpy(float)
    k = np.rint(n*p).astype(int)
    if not np.allclose(k, n*p, atol=1e-7):
        raise ValueError("Noninteger coverage counts")
    lo, hi = np.zeros(len(k)), np.ones(len(k))
    use = k > 0
    lo[use] = beta.ppf(.025, k[use], n[use]-k[use]+1)
    use = k < n
    hi[use] = beta.ppf(.975, k[use]+1, n[use]-k[use])
    step["coverage_mc_lower"], step["coverage_mc_upper"] = lo, hi
    return step


def add_baseline_to_comparisons(frame):
    """Display each once-simulated baseline in matched predictor regimes.

    This is a plotting view only: never pool these copies as independent trials.
    """
    baseline = frame[frame.method.isin(BASELINE_METHODS)]
    other = frame[~frame.method.isin(BASELINE_METHODS)]
    if baseline.empty or other.empty:
        return frame
    pieces = [other]
    matched = ["theta_star", "z_sha256", "n_questions", "horizon", "alpha"]
    for scenario, group in baseline.groupby("scenario"):
        targets = other[other.scenario == scenario]
        if targets.empty:
            pieces.append(group)
            continue
        for regime, comparison in targets.groupby("regime"):
            for key in matched:
                if pd.concat([group[key], comparison[key]]).nunique(dropna=False) != 1:
                    raise ValueError(f"Hedged-CS baseline and {scenario}/{regime} differ in {key}")
            pieces.append(group.assign(regime=regime))
    return pd.concat(pieces, ignore_index=True)


def plot_results(root, output, anchor_step=1000, *, show_repeat_counts=False):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    step, stop, sources = load_results(root)
    step = coverage_intervals(step)
    step.to_csv(output/"step_summary.csv", index=False)
    stop.to_csv(output/"stopping_summary.csv", index=False)
    terminal = step.sort_values("step").groupby(KEYS, as_index=False).tail(1)
    terminal.to_csv(output/"validity_summary.csv", index=False)
    if "model_row" in step:
        step.loc[step.model_row.notna(), ["scenario", "theta_star"]+M2_METADATA].drop_duplicates().to_csv(
            output/"model_catalog.csv", index=False)
    baseline_step = step[step.method.isin(BASELINE_METHODS)].copy()
    step, stop = add_baseline_to_comparisons(step), add_baseline_to_comparisons(stop)
    terminal = step.sort_values("step").groupby(KEYS, as_index=False).tail(1)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "pdf.fonttype": 42})
    files = []

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(output/f"{name}.{ext}", dpi=240, bbox_inches="tight")
        plt.close(fig)
        files.append(name)

    cases = list(step.groupby(["scenario", "regime"], sort=False).groups)

    def case_title(scenario, regime, separator=" / "):
        frame = step[(step.scenario == scenario) & (step.regime == regime)]
        if "model_row" in frame and frame.model_row.notna().all():
            return f"M2 row {int(frame.model_row.iloc[0])}{separator}{regime}"
        label = scenario_label(scenario)
        return f"{label}{separator}{regime}"

    def count_label(label, frame):
        if not show_repeat_counts:
            return label
        low, high = int(frame.n_repeats.min()), int(frame.n_repeats.max())
        count = str(low) if low == high else f"{low}–{high}"
        return f"{label} (R={count})"

    def panels(panel_cases=None, paired=False, vertical=False):
        panel_cases = cases if panel_cases is None else panel_cases
        cols = 1 if vertical else min(2, len(panel_cases))
        rows = math.ceil(len(panel_cases)/cols)
        figsize = ((4.4, 3.6*rows) if vertical else (8.4, 3.6)) if paired else (3.6*cols, 2.8*rows)
        fig, axes = plt.subplots(rows, cols, sharey=paired,
                                 figsize=figsize, squeeze=False)
        for ax in axes.flat[len(panel_cases):]:
            ax.set_visible(False)
        return fig, axes.flat

    def line_figure(frame, xcol, metric, ylabel, name, coverage=False, stopping=False,
                    panel_cases=None, paired=False, vertical=False):
        panel_cases = cases if panel_cases is None else panel_cases
        fig, axes = panels(panel_cases, paired=paired, vertical=vertical)
        for ax, (scenario, regime) in zip(axes, panel_cases):
            f = frame[(frame.scenario == scenario) & (frame.regime == regime)]
            for method, (label, color, marker) in METHODS.items():
                g = f[f.method == method].dropna(subset=[f"{metric}_mean"]).sort_values(xcol)
                if g.empty:
                    continue
                x, y = g[xcol].to_numpy(), g[f"{metric}_mean"].to_numpy()
                se = g[f"{metric}_se"].fillna(0).to_numpy()
                policy = method.rsplit("_", 1)[-1]
                line_marker = marker if stopping or policy in ("maxmin", "maxmax") else None
                marker_positions = np.linspace(0, len(x)-1, min(12, len(x)), dtype=int)
                ax.plot(x, y, color=color, label=count_label(label, g), lw=1.4,
                        linestyle="-",
                        marker=line_marker, markevery=marker_positions, markersize=5)
                if coverage:
                    ax.fill_between(x, g.coverage_mc_lower.to_numpy(), g.coverage_mc_upper.to_numpy(),
                                    color=color, alpha=.12)
                else:
                    ax.fill_between(x, np.maximum(0, y-se), y+se, color=color, alpha=.12)
                if stopping:
                    for xx, yy, hit in zip(x, y, g.reached_mean):
                        ax.plot(xx, yy, marker=marker, color=color,
                                markerfacecolor=color if np.isclose(hit, 1) else "white", ms=5)
            ax.set_title(case_title(scenario, regime))
            ax.set_xlabel("Width threshold $\\epsilon$" if stopping else "Queries $t$")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
            if coverage:
                ax.axhline(1-f.alpha.iloc[0], color="black", ls=":", lw=1)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.set_ylim(max(0, f.coverage_mc_lower.min()-.015), 1.01)
            else:
                ax.set_ylim(bottom=0)
            if paired:
                ax.legend(loc="upper right", fontsize=7)
                ax.tick_params(axis="y", labelleft=True)
                ax.set_xticks(sorted(f[xcol].unique()))
            else:
                ax.legend(fontsize=7)
        fig.tight_layout()
        save(fig, name)

    for metric, ylabel, name in (
        ("width", "Mean nested CS full width", "width"),
        ("active_candidates", "Mean surviving candidates", "active_candidates"),
        ("empty_cs", "Fraction with empty CS", "empty_cs"),
        ("mae_z", "Mean pre-query MAE", "prediction_mae"),
        ("brier_z", "Mean pre-query Brier error", "prediction_brier"),
        ("cum_kl_z_query", "Mean cumulative query-weighted KL", "cumulative_kl"),
    ):
        if step[f"{metric}_mean"].notna().any():
            line_figure(step, "step", metric, ylabel, name)
    line_figure(step, "step", "anytime_covered", "Coverage at all steps through $t$",
                "validity_through_t", coverage=True)
    line_figure(stop, "epsilon", "stop_time_capped", "Mean capped stopping queries",
                "stopping_time", stopping=True)
    for index in (1, 2, 3):
        group = f"z_{index}"
        regimes = sorted({r for s, r in cases if s in (group, f"tilde_{group}")})
        for regime in regimes:
            pair = [(group, regime), (f"tilde_{group}", regime)]
            if not all(case in cases for case in pair):
                continue
            suffix = f"_{regime}" if len(regimes) > 1 else ""
            line_figure(stop, "epsilon", "stop_time_capped", "Mean capped stopping queries",
                        f"stopping_time_{group}{suffix}", stopping=True,
                        panel_cases=pair, paired=True)
            if index == 1:
                line_figure(stop, "epsilon", "stop_time_capped", "Mean capped stopping queries",
                            f"stopping_time_{group}{suffix}_vertical", stopping=True,
                            panel_cases=pair, paired=True, vertical=True)

    fig, ax = plt.subplots(figsize=(max(5.5, len(cases)*1.5), 3.3))
    present = [m for m in METHODS if (terminal.method == m).any()]
    for j, method in enumerate(present):
        label, color, marker = METHODS[method]
        g = terminal[terminal.method == method]
        if g.empty:
            continue
        x = (np.array([cases.index((r.scenario, r.regime)) for r in g.itertuples()])
             + (j-(len(present)-1)/2)*.16)
        y = g.anytime_covered_mean.to_numpy()
        ax.errorbar(x, y, yerr=np.vstack((np.maximum(0, y-g.coverage_mc_lower.to_numpy()),
                                        np.maximum(0, g.coverage_mc_upper.to_numpy()-y))),
                    color=color, fmt=marker, capsize=3, label=count_label(label, g))
    if step.alpha.nunique() != 1:
        raise ValueError("Plot one alpha at a time")
    ax.axhline(1-step.alpha.iloc[0], color="black", ls=":")
    ax.set_xticks(range(len(cases)), [case_title(s, r, "\n") for s, r in cases], fontsize=8)
    ax.set_ylabel("Coverage at all steps through horizon")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(max(0, terminal.coverage_mc_lower.min()-.015), 1.01)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "validity_summary")

    # These are rate guides, NOT a full numerical bound or a fitted decay law.
    oracle = step[step.predictor == "oracle"]
    diagnostics = []
    kappa_colors = {k: plt.cm.tab10.colors[j % 10] for j, k in enumerate(sorted(oracle.kappa.unique()))}

    def draw_theory_panel(ax, subset, family, scenario, policy, cumulative, *, record_diagnostics=False):
        for kappa, g in subset.groupby("kappa"):
            color = kappa_colors[kappa]
            g = g.sort_values("step")
            t = g.step.to_numpy(float)
            metric = ("cum_kl_z_query" if family == "ripr" else "cum_mae_z") if cumulative else "width"
            y = g[f"{metric}_mean"].to_numpy()
            se = g[f"{metric}_se"].fillna(0).to_numpy()
            reference, rate_label = (cumulative_reference(t, kappa) if cumulative
                                     else rate_reference(t, kappa, family))
            valid = (y > 0) & np.isfinite(y)
            ax.loglog(t[valid], y[valid], color=color,
                      label=rf"$\kappa={kappa:g}$: ${rate_label}$")
            ax.fill_between(t[valid], np.maximum(y[valid]-se[valid], 1e-12), y[valid]+se[valid],
                            color=color, alpha=.12)
            effective_anchor = anchor_step if t[-1] >= anchor_step else max(1, int(t[-1]//5))
            positions = np.flatnonzero(valid & (t >= effective_anchor))
            if len(positions):
                at = positions[0]
                guide = reference*y[at]/reference[at]
                ax.loglog(t, guide, color=color, ls="--", lw=1)
                if not cumulative and record_diagnostics:
                    fit = valid & (t >= t[at])
                    slope = float(np.polyfit(np.log(t[fit]), np.log(y[fit]), 1)[0]) if fit.sum() > 1 else None
                    diagnostics.append(dict(scenario=scenario, method=f"{family}_{policy}",
                                            kappa=kappa, anchor_step=int(t[at]), rate=rate_label,
                                            empirical_loglog_slope=slope,
                                            theoretical_mismatch="KL" if family == "ripr" else "MAE",
                                            terminal_empty_fraction=float(g.empty_cs_mean.iloc[-1])))
        if not cumulative:
            for baseline_method in BASELINE_METHODS:
                reference = baseline_step[(baseline_step.scenario == scenario)
                                          & (baseline_step.method == baseline_method)].sort_values("step")
                if reference.empty:
                    continue
                label, color, _ = METHODS[baseline_method]
                valid = reference.width_mean > 0
                ax.loglog(reference.step[valid], reference.width_mean[valid],
                          color=color, label=label, lw=1.4)
                y = reference.width_mean[valid].to_numpy()
                se = reference.width_se[valid].fillna(0).to_numpy()
                ax.fill_between(reference.step[valid].to_numpy(), np.maximum(y-se, 1e-12),
                                y+se, color=color, alpha=.12)
        if cumulative:
            ax.set_title(f"{scenario} / {policy}")
        ax.set_xlabel("Queries $t$ (log scale)")
        ax.set_ylabel(("Cumulative KL" if family == "ripr" else "Cumulative MAE")
                      if cumulative else
                      f"Mean {'RIPr' if family == 'ripr' else 'Betting'}-CS width (log scale)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)

    for family in ("ripr", "bet"):
        f = oracle[oracle.method.str.startswith(family+"_")]
        if f.empty:
            continue
        scenarios = list(f.scenario.unique())
        policies = [p for p in QUERY_POLICIES if (f.method == f"{family}_{p}").any()]
        for cumulative in (False, True):
            fig, axes = plt.subplots(len(scenarios), len(policies), squeeze=False,
                                     figsize=(3.8*len(policies), 3.3*len(scenarios)))
            for row, scenario in enumerate(scenarios):
                for col, policy in enumerate(policies):
                    subset = f[(f.scenario == scenario) & (f.method == f"{family}_{policy}")]
                    draw_theory_panel(axes[row, col], subset, family, scenario, policy, cumulative,
                                      record_diagnostics=True)
            if cumulative:
                title = f"{'RIPr' if family == 'ripr' else 'Betting'}: shared predictor $a_s=a_0s^{{-\\kappa}}$"
                fig.suptitle(title)
            fig.tight_layout(rect=(0, 0, 1, .95 if cumulative else 1))
            save(fig, f"{'cumulative_theory_mismatch' if cumulative else 'width_vs_theory_rate'}_{family}")

    # The same curves and rate guides, with RIPr on the left and betting on the right.
    paired_cases = [(scenario, policy) for scenario in oracle.scenario.unique()
                    for policy in QUERY_POLICIES
                    if all(((oracle.scenario == scenario) & (oracle.method == f"{family}_{policy}")).any()
                           for family in ("ripr", "bet"))]
    if paired_cases:
        fig, axes = plt.subplots(len(paired_cases), 2, squeeze=False, sharex="row", sharey="row",
                                 figsize=(9.2, 3.8*len(paired_cases)))
        for row, (scenario, policy) in enumerate(paired_cases):
            for col, family in enumerate(("ripr", "bet")):
                subset = oracle[(oracle.scenario == scenario) & (oracle.method == f"{family}_{policy}")]
                draw_theory_panel(axes[row, col], subset, family, scenario, policy, False)
                axes[row, col].tick_params(axis="y", labelleft=True)
        fig.tight_layout()
        save(fig, "width_vs_theory_rate_combined")
    pd.DataFrame(diagnostics, columns=["scenario", "method", "kappa", "anchor_step", "rate",
                                      "empirical_loglog_slope", "theoretical_mismatch", "terminal_empty_fraction"]
                 ).to_csv(output/"rate_diagnostics.csv", index=False)
    write_json(output/"plot_manifest.json", dict(
        sources=sources, figures=files, coverage="exact candidate membership, through all t",
        uncertainty="pointwise 95% Clopper-Pearson for coverage; +/-1 SE otherwise",
        footnotes_shown=False,
        repeat_counts_in_legends=show_repeat_counts,
        method_style="family color: RIPr purple, betting orange; policy: uniform solid, max-min solid with square markers, max-max solid with X markers; Hedged baselines retain their own colors",
        stopping="mean min(stopping time,horizon); hollow markers indicate some repeats did not reach a nonempty target",
        empty_cs="empty-set diameter is zero but does not count as a precision hit; see empty_cs figures",
        hedged_baseline="one run per scenario per baseline: hedged_uniform (WR), hedged_wor (WoR); reused across predictor regimes only for display; no predictor mismatch metric",
        hedged_wor="Theorem 4 plug-in bets plus deterministic feasibility bounds; census is a singleton or remains empty if the truth was already rejected; zero widths omitted on log axes",
        theory="fixed-parameter rate guides, normalized at recorded anchors; not numerical bounds",
        betting_caveat="MAE mismatch; retains sqrt(t)/t term; half-range proof differs from full-range implementation",
        real_data="m2_model_bank_v1 uses unaltered selected M2 answers; see model_catalog.csv. z_1 contains the prepared M2 answers; tilde_z_1 is a permuted control"))
    print(f"Saved {len(files)} PNG/PDF figure pairs: {output.resolve()}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare-m2", help="Prepare explicit M2 rows, or the model farthest from a reference")
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--m2-path", required=True, help="External original M2.csv; not needed for bundled datasets")
    prep.add_argument("--model-rows", nargs="+", type=nonnegative_int,
                      help="Explicit zero-based rows; omit to select the farthest model")
    prep.add_argument("--predictor-snapshot", default=str(ROOT/"data/benchmark_5000/predictor.npz"),
                      help="NPZ containing v, mu and sigma aligned with the first N M2 questions")
    prep.add_argument("--base-dataset", default="mmlu-pro")
    prep.add_argument("--reference-row", type=nonnegative_int, default=1,
                      help="Zero-based M2 row to exclude and compare accuracy against")
    prep.add_argument("--n-questions", type=positive_int, default=5000)
    run_parser = commands.add_parser("run", help="Simulate nested CSs on prepared fixed labels")
    run_parser.add_argument("--data-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--scenarios", nargs="+", type=canonical_scenario,
                            help="z_1 z_2 z_3 tilde_z_1 tilde_z_2 tilde_z_3, or prepared M2 names")
    run_parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(DEFAULT_METHODS),
                            help="Methods to run; max-max variants are opt-in")
    run_parser.add_argument("--predictor", choices=("faq", "oracle", "custom"), default="faq")
    run_parser.add_argument("--predictor-factory", help="Python file:factory, receiving one context dictionary")
    run_parser.add_argument("--predictor-kwargs", default="{}", help="JSON parameters passed to custom factory")
    run_parser.add_argument("--custom-oracle", action="store_true", help="Explicitly give full z to custom factory")
    run_parser.add_argument("--kappas", nargs="+", type=float, default=[0., .5, 1.])
    run_parser.add_argument("--oracle-amplitude", type=float, default=.9)
    run_parser.add_argument("--oracle-center", type=float, default=.7)
    run_parser.add_argument("--n-repeats", type=positive_int, default=10)
    run_parser.add_argument("--repeat-start", type=nonnegative_int, default=0)
    run_parser.add_argument("--repeat-batch-size", type=positive_int, default=8)
    run_parser.add_argument("--candidate-batch-size", type=positive_int, default=256)
    run_parser.add_argument("--max-steps", type=positive_int)
    run_parser.add_argument("--seed", type=nonnegative_int, default=0)
    run_parser.add_argument("--alpha", type=float, default=.05)
    run_parser.add_argument("--hedge-cap", type=float, default=.5,
                            help="Hedged-CS c in eq. (25); default 0.5")
    run_parser.add_argument("--hedge-weight", type=float, default=.5,
                            help="Weight on K_plus in the hedged max; default 0.5")
    run_parser.add_argument("--epsilons", nargs="+", type=float, default=list(DEFAULT_EPSILONS))
    run_parser.add_argument("--uniform-weight", type=float, default=.05)
    run_parser.add_argument("--grow-opt-steps", type=nonnegative_int, default=25)
    run_parser.add_argument("--grow-opt-lr", type=float, default=.5)
    run_parser.add_argument("--grow-opt-rtol", type=float, default=OPT_RTOL)
    run_parser.add_argument("--grow-opt-atol", type=float, default=OPT_ATOL)
    run_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run_parser.add_argument("--threads", type=positive_int, default=1)
    run_parser.add_argument("--progress-every", type=positive_int, default=100)
    run_parser.add_argument("--anchor-step", type=positive_int, default=1000)
    run_parser.add_argument("--no-plots", action="store_true")
    plot_parser = commands.add_parser("plot", help="Plot one run or recursively merge completed run parts")
    plot_parser.add_argument("--results-dir", required=True)
    plot_parser.add_argument("--output-dir")
    plot_parser.add_argument("--anchor-step", type=positive_int, default=1000)
    plot_parser.add_argument("--show-repeat-counts", action="store_true",
                             help="Include available repeat counts in comparison plot legends")
    export_parser = commands.add_parser("export", help="Create compact plotting data from completed run(s)")
    export_parser.add_argument("--results-dir", required=True, nargs="+",
                               help="One or more directories containing disjoint completed runs")
    export_parser.add_argument("--output-dir", required=True, help="Root data directory to create/update")
    export_parser.add_argument("--experiment", required=True,
                               choices=("comparison", "validity", "mismatch", "m2_comparison"))
    return parser.parse_args(argv)

if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "prepare-m2":
        prepare_m2(arguments)
    elif arguments.command == "run":
        run(arguments)
    elif arguments.command == "export":
        export_results(arguments)
    else:
        plot_results(arguments.results_dir, arguments.output_dir or Path(arguments.results_dir)/"plots",
                     arguments.anchor_step, show_repeat_counts=arguments.show_repeat_counts)
