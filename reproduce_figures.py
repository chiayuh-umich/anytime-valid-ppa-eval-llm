"""Reproduce numbered paper figures and diagnostics from bundled summaries.

NumPy, pandas, Matplotlib and SciPy suffice. PyTorch is imported only when
--*-results points to newly simulated run directories instead of bundled CSVs.
The output root contains main_figure/, appendix_figure/ and figures/.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nested-reproduction-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent
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
BASELINES = ("hedged_uniform", "hedged_wor")
M2_SCENARIOS = tuple(f"m2_row_{row:04d}" for row in (1183, 1506, 483, 542))
BBH_M2_SCENARIOS = tuple(f"m2_row_{row:04d}" for row in (788, 1752, 1516, 729))
EXPERIMENTS = ("comparison", "validity", "mismatch", "m2_comparison", "validity_r800",
               "bbh_m2_comparison")
# Fixed paper numbering, independent of which files already exist on disk.
PAPER_FIGURES = {
    "width_vs_theory_rate_combined": "main_figure/1_width_vs_theory_rate",
    "stopping_time_z123": "main_figure/2_stopping_time_z123",
    "width_z123": "appendix_figure/3_CS_width_z123",
    "one_step_mae_z123": "appendix_figure/4_one_step_mae_z123",
    "one_step_kl_query_z123": "appendix_figure/5_one_step_kl_query_z123",
    "stopping_time_paired_z123": "appendix_figure/6_stopping_time_paired_z123",
    "stopping_time_m2_comparison": "appendix_figure/7_stopping_time_m2_comparison",
    "one_step_mae_m2_comparison": "appendix_figure/8_one_step_mae_m2_comparison",
    "validity_through_t": "appendix_figure/9_validity_through_t",
    "validity_tilde_z2_bet_uniform_r800": "appendix_figure/10_validity_tilde_z2_bet_uniform_r800",
    "stopping_time_bbh_m2_comparison": "appendix_figure/stopping_time_bbh_m2_comparison",
    "one_step_kl_query_bbh_m2_comparison": "appendix_figure/one_step_kl_query_bbh_m2_comparison",
    "one_step_mae_bbh_m2_comparison": "appendix_figure/one_step_mae_bbh_m2_comparison",
}
VALIDITY_LABELS = {
    "z_2": r"$z_2$",
    "z_3": r"$z_3$",
    "tilde_z_2": r"$\tilde{z}_2$",
    "tilde_z_3": r"$\tilde{z}_3$",
}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_data(directory):
    """Check the distributed data before plotting; failures are never ignored."""
    path = directory / "checksums.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing data checksums: {path}")
    checksums = json.loads(path.read_text())
    for name, expected in checksums.items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Data checksum mismatch: {name}")
    print(f"Verified {len(checksums)} data files.", flush=True)


def read_summaries(directory, new_results=None):
    if new_results is not None:
        from nested_experiment import load_results
        step, stop, sources = load_results(new_results)
        roots = ([Path(p).resolve() for p in new_results]
                 if isinstance(new_results, (list, tuple)) else [Path(new_results).resolve()])
        hashes = {}
        for source in sources:
            path = Path(source)
            root = next(p for p in roots if path.is_relative_to(p))
            key = (Path(root.name) / path.relative_to(root)).as_posix()
            if key in hashes:
                raise ValueError(f"Ambiguous source provenance: {key}")
            hashes[key] = file_hash(path)
        return step, stop, hashes
    paths = [directory / f"{kind}_summary.csv.gz" for kind in ("step", "stopping")]
    # Preserve the full precision written by pandas when packing the summaries.
    step, stop = [pd.read_csv(p, float_precision="round_trip") for p in paths]
    for frame, xcol in ((step, "step"), (stop, "epsilon")):
        if frame.duplicated(KEYS + [xcol]).any():
            raise ValueError(f"Duplicate summary rows in {directory}")
    return step, stop, {str(p.relative_to(directory.parent)): file_hash(p) for p in paths}


def select_faq(frame, scenarios):
    frame = frame[frame.scenario.isin(scenarios) & (
        frame.regime.eq("faq") | frame.method.isin(BASELINES))].copy()
    if set(frame.scenario) != set(scenarios):
        raise ValueError("All requested datasets must have completed results")
    for column in ("horizon", "alpha", "n_questions"):
        if frame[column].nunique() != 1:
            raise ValueError(f"Comparisons require a common {column}")
    for scenario, group in frame.groupby("scenario"):
        if group.z_sha256.nunique() != 1:
            raise ValueError(f"Different ground truths for {scenario}")
    return frame


def figure_path(output, stem):
    """Plot helpers receive the figures/ directory used for diagnostic tables."""
    return output.parent / PAPER_FIGURES[stem] if stem in PAPER_FIGURES else output / stem


def save_figure(fig, output, stem):
    target = figure_path(output, stem)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        fig.savefig(target.with_suffix(f".{ext}"), dpi=240, bbox_inches="tight", metadata=metadata)
    plt.close(fig)
    print(f"Saved {target.relative_to(output.parent)}.png / .pdf", flush=True)
    return stem


def plot_comparison(step, stop, output, kind, paired=False):
    """Stopping, width, MAE or query-weighted KL; three or six panels."""
    panels = [(f"{prefix}z_{i}", r"\tilde{z}" if prefix else "z", i)
              for i in (1, 2, 3)
              for prefix in (("", "tilde_") if paired else ("",))]
    selected = select_faq(stop if kind == "stopping" else step, [p[0] for p in panels])
    settings = {
        "stopping": ("epsilon", "stop_time_capped", "Mean capped stopping queries", "stopping_time"),
        "width": ("step", "width", "Mean CS width", "width"),
        "mae": ("step", "mae_z", "Mean one-step MAE mismatch", "one_step_mae"),
        "kl": ("step", "kl_z_query", "Mean query-weighted KL mismatch", "one_step_kl_query"),
    }
    xcol, metric, ylabel, prefix = settings[kind]
    if kind in ("mae", "kl"):
        selected = selected[selected.method.str.startswith("bet_" if kind == "mae" else "ripr_")]
    if selected.duplicated(["scenario", "method", xcol]).any():
        raise ValueError("Ambiguous plotted regimes")
    shape = (3, 2) if paired else (1, 3)
    fig, axes = plt.subplots(*shape, figsize=(8.4, 10.8) if paired else (12.6, 3.6),
                             sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for (scenario, symbol, i), ax in zip(panels, axes):
        frame = selected[selected.scenario == scenario]
        for method, (label, color, marker) in METHODS.items():
            g = frame[frame.method == method].sort_values(xcol)
            if g.empty:
                continue
            x, y = g[xcol].to_numpy(), g[f"{metric}_mean"].to_numpy()
            se = g[f"{metric}_se"].fillna(0).to_numpy()
            count = 10 if kind in ("mae", "kl") else 12
            positions = None if kind == "stopping" else np.linspace(0, len(x)-1, min(count, len(x)), dtype=int)
            ax.plot(x, y, color=color, marker=marker, markevery=positions,
                    linestyle="-", lw=1.4, ms=5, label=label)
            ax.fill_between(x, np.maximum(0, y-se), y+se, color=color, alpha=.12)
            if kind == "stopping":
                for xx, yy, hit in zip(x, y, g.reached_mean):
                    if not np.isclose(hit, 1):
                        ax.plot(xx, yy, linestyle="none", marker=marker, color=color,
                                markerfacecolor="white", ms=5)
        ax.set_title(rf"${symbol}_{{{i}}}$")
        ax.set_xlabel(r"Width threshold $\epsilon$" if kind == "stopping" else
                      r"Query step $t$" if kind in ("mae", "kl") else r"Queries $t$", fontsize=12)
        if kind == "stopping":
            ax.set_xticks(sorted(selected.epsilon.unique()))
        ax.tick_params(axis="both", labelsize=11, labelleft=True, labelbottom=True)
        ax.grid(alpha=.2)
        ax.legend(loc="upper right", fontsize=8, markerscale=1.25)
    # Only fix shared limits after all panels have contributed their data.
    if kind in ("mae", "kl"):
        upper = (selected[f"{metric}_mean"] + selected[f"{metric}_se"].fillna(0)).max()
        axes[0].set_ylim(0, max(float(upper)*1.05, 1e-12))
    else:
        axes[0].set_ylim(bottom=0)
    for ax in axes[::shape[1]]:
        ax.set_ylabel(ylabel, fontsize=12)
    return save_figure(fig, output, f"{prefix}_{'paired_' if paired else ''}z123")


def coverage_intervals(step):
    step = step.copy()
    n, p = step.n_repeats.to_numpy(int), step.anytime_covered_mean.to_numpy(float)
    k = np.rint(n*p).astype(int)
    if not np.allclose(k, n*p, atol=1e-7):
        raise ValueError("Noninteger Monte Carlo coverage counts")
    lo, hi = np.zeros(len(k)), np.ones(len(k))
    use = k > 0
    lo[use] = beta.ppf(.025, k[use], n[use]-k[use]+1)
    use = k < n
    hi[use] = beta.ppf(.975, k[use]+1, n[use]-k[use])
    step["coverage_mc_lower"], step["coverage_mc_upper"] = lo, hi
    return step


def plot_m2_comparison(step, stop, output, scenarios=M2_SCENARIOS,
                       kinds=None, suffix="m2_comparison"):
    """Four matched M2 models; all methods remain separate in each model panel."""
    step, stop = (select_faq(frame, scenarios) for frame in (step, stop))
    for frame, xcol in ((step, "step"), (stop, "epsilon")):
        if frame.duplicated(["scenario", "method", xcol]).any():
            raise ValueError("Ambiguous M2 regimes; expected one regime per method")
        for column in ("n_repeats", "predictor_snapshot_sha256", "question_columns_sha256"):
            if column not in frame or frame[column].isna().any() or frame[column].nunique() != 1:
                raise ValueError(f"M2 comparisons require matched {column}")
        expected_x = np.arange(1, int(frame.horizon.iloc[0]) + 1) if xcol == "step" else np.array(
            sorted(frame.epsilon.unique()))
        for scenario, group in frame.groupby("scenario"):
            if set(group.method) != set(METHODS):
                raise ValueError(f"Missing M2 methods for {scenario}")
            for method, values in group.groupby("method"):
                if not np.array_equal(values.sort_values(xcol)[xcol].to_numpy(), expected_x):
                    raise ValueError(f"Incomplete {xcol} grid for {scenario}/{method}")
    for scenario in scenarios:
        s, t = step[step.scenario == scenario], stop[stop.scenario == scenario]
        for column in ("z_sha256", "theta_star", "n_repeats", "n_questions", "horizon", "alpha"):
            if pd.concat([s[column], t[column]]).nunique() != 1:
                raise ValueError(f"Step and stopping data disagree on {column} for {scenario}")

    step = coverage_intervals(step)
    ripr_steps = step[step.method.str.startswith("ripr_")]
    bet_steps = step[step.method.str.startswith("bet_")]
    settings = {
        "stopping": (stop, "epsilon", "stop_time_capped", "Mean capped stopping queries",
                     "stopping_time_m2_comparison"),
        "width": (step, "step", "width", "Mean CS width", "width_m2_comparison"),
        "coverage": (step, "step", "anytime_covered", r"Coverage at all steps through $t$",
                     "coverage_m2_comparison"),
        "kl": (ripr_steps, "step", "kl_z_query", "Mean query-weighted KL mismatch",
               "one_step_kl_query_m2_comparison"),
        "mae": (bet_steps, "step", "mae_z", "Mean one-step MAE mismatch",
                "one_step_mae_m2_comparison"),
        "cum_kl": (ripr_steps, "step", "cum_kl_z_query", "Mean cumulative KL mismatch",
                   "cumulative_kl_query_m2_comparison"),
        "cum_mae": (bet_steps, "step", "cum_mae_z", "Mean cumulative MAE mismatch",
                    "cumulative_mae_m2_comparison"),
    }
    files = []
    for kind, (selected, xcol, metric, ylabel, stem) in settings.items():
        if kinds is not None and kind not in kinds:
            continue
        stem = stem.removesuffix("m2_comparison") + suffix
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True, sharey=True)
        for ax, scenario in zip(axes.flat, scenarios):
            frame = selected[selected.scenario == scenario]
            row, theta = int(frame.model_row.iloc[0]), float(frame.theta_star.iloc[0])
            for method, (label, color, marker) in METHODS.items():
                g = frame[frame.method == method].sort_values(xcol)
                if g.empty:
                    continue
                x, y = g[xcol].to_numpy(), g[f"{metric}_mean"].to_numpy()
                positions = None if kind == "stopping" else np.linspace(0, len(x)-1, 11, dtype=int)
                ax.plot(x, y, color=color, marker=marker, markevery=positions,
                        linestyle="-", lw=1.6, ms=5.5, label=label)
                if kind == "coverage":
                    lower, upper = g.coverage_mc_lower.to_numpy(), g.coverage_mc_upper.to_numpy()
                else:
                    se = g[f"{metric}_se"].fillna(0).to_numpy()
                    lower, upper = np.maximum(0, y-se), y+se
                ax.fill_between(x, lower, upper, color=color, alpha=.12)
                if kind == "stopping":
                    for xx, yy, hit in zip(x, y, g.reached_mean):
                        if not np.isclose(hit, 1):
                            ax.plot(xx, yy, linestyle="none", marker=marker, color=color,
                                    markerfacecolor="white", ms=5.5)
            ax.set_title(rf"M2 row {row}: $\theta^\star={theta:.4f}$", fontsize=14)
            ax.set_xlabel(r"Width threshold $\epsilon$" if kind == "stopping" else r"Queries $t$",
                          fontsize=13)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.tick_params(labelsize=11, labelleft=True, labelbottom=True)
            ax.grid(alpha=.2)
            ax.legend(loc="lower left" if kind == "coverage" else "upper right",
                      fontsize=9, markerscale=1.2)
            if kind == "stopping":
                ax.set_xticks(sorted(selected.epsilon.unique()))
            elif kind == "coverage":
                ax.axhline(1-float(frame.alpha.iloc[0]), color="black", ls=":", lw=1)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
        if kind == "coverage":
            axes.flat[0].set_ylim(max(0, float(selected.coverage_mc_lower.min())-.015), 1.01)
        else:
            axes.flat[0].set_ylim(bottom=0)
        files.append(save_figure(fig, output, stem))

    order = {s: i for i, s in enumerate(scenarios)}
    stopping_values = stop.sort_values(["scenario", "method", "epsilon"], key=lambda c:
                                      c.map(order) if c.name == "scenario" else c)
    stopping_values.to_csv(output / f"stopping_time_{suffix}_values.csv", index=False)
    if kinds is None or "coverage" in kinds:
        terminal = step.sort_values("step").groupby(KEYS, as_index=False).tail(1)
        terminal.to_csv(output / f"coverage_{suffix}_terminal.csv", index=False)
    return files


def plot_validity(step, output):
    step = coverage_intervals(select_faq(step, VALIDITY_LABELS))
    if step.duplicated(["scenario", "method", "step"]).any():
        raise ValueError("Multiple validity regimes per method")
    if step.n_repeats.nunique() != 1:
        raise ValueError("Validity panels require equal repeat counts")
    methods = [m for m in METHODS if step.method.eq(m).any()]
    for scenario, group in step.groupby("scenario"):
        if set(group.method) != set(methods):
            raise ValueError(f"Missing validity methods for {scenario}")
    terminal = step.sort_values("step").groupby(KEYS, as_index=False).tail(1)
    alpha = float(step.alpha.iloc[0])
    ymin = max(0, float(step.coverage_mc_lower.min()) - .015)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6), sharex=True, sharey=True)
    for ax, (scenario, label) in zip(axes.flat, VALIDITY_LABELS.items()):
        for method in methods:
            legend, color, marker = METHODS[method]
            g = step[(step.scenario == scenario) & (step.method == method)].sort_values("step")
            x = g.step.to_numpy()
            positions = np.linspace(0, len(x)-1, min(10, len(x)), dtype=int)
            ax.plot(x, g.anytime_covered_mean.to_numpy(), color=color, linestyle="-",
                    marker=marker, markevery=positions, ms=4, lw=1.3, label=legend)
            ax.fill_between(x, g.coverage_mc_lower.to_numpy(), g.coverage_mc_upper.to_numpy(),
                            color=color, alpha=.12)
        ax.set_title(label, fontsize=15)
        ax.axhline(1-alpha, color="black", ls=":", lw=1)
        ax.set_ylim(ymin, 1.01)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel(r"Queries $t$", fontsize=12)
        ax.set_ylabel(r"Coverage at all steps through $t$", fontsize=11)
        ax.tick_params(labelsize=10, labelleft=True, labelbottom=True)
        ax.grid(alpha=.2)
        ax.legend(loc="lower left", fontsize=7, markerscale=1.15)
    files = [save_figure(fig, output, "validity_through_t")]

    fig, ax = plt.subplots(figsize=(7, 3.7))
    scenarios = list(VALIDITY_LABELS)
    for j, method in enumerate(methods):
        legend, color, marker = METHODS[method]
        g = terminal[terminal.method == method].set_index("scenario").loc[scenarios]
        x = np.arange(len(scenarios)) + (j-(len(methods)-1)/2)*.12
        y = g.anytime_covered_mean.to_numpy()
        error = np.vstack((np.maximum(0, y-g.coverage_mc_lower.to_numpy()),
                           np.maximum(0, g.coverage_mc_upper.to_numpy()-y)))
        ax.errorbar(x, y, yerr=error, fmt=marker, color=color, capsize=3, label=legend)
    ax.axhline(1-alpha, color="black", ls=":", lw=1)
    ax.set_ylim(ymin, 1.01)
    ax.set_xticks(range(len(scenarios)), list(VALIDITY_LABELS.values()), fontsize=14)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Coverage at all steps through horizon", fontsize=11)
    ax.legend(loc="lower left", fontsize=8)
    files.append(save_figure(fig, output, "validity_summary"))
    terminal.to_csv(output / "validity_terminal_values.csv", index=False)
    return files


def plot_validity_r800(step, output, repeat_coverage=None):
    """Standalone follow-up; never pool its repeats with the original 300."""
    step = coverage_intervals(select_faq(step, ("tilde_z_2",))).sort_values("step")
    if set(step.method) != {"bet_uniform"} or set(step.regime) != {"faq"}:
        raise ValueError("The standalone validity plot requires only FAQ / betting uniform")
    for column in ("n_repeats", "horizon", "alpha", "n_questions"):
        if step[column].nunique() != 1:
            raise ValueError(f"Standalone validity requires a common {column}")
    horizon, repeats = int(step.horizon.iloc[0]), int(step.n_repeats.iloc[0])
    x = step.step.to_numpy(int)
    if not np.array_equal(x, np.arange(1, horizon + 1)):
        raise ValueError("Standalone validity needs exactly one row for every step")
    counts = np.rint(repeats * step.anytime_covered_mean.to_numpy()).astype(int)
    if np.any((counts < 0) | (counts > repeats)) or np.any(np.diff(counts) > 0):
        raise ValueError("Invalid through-time coverage counts")
    if repeat_coverage is not None:
        records = pd.read_csv(repeat_coverage)
        first = records.first_exclusion_step.to_numpy(float)
        if (len(records) != repeats or records.repeat_id.duplicated().any()
                or not records.horizon.eq(horizon).all()
                or not np.all(np.isfinite(first) & (first == np.floor(first)))
                or not np.all((first == -1) | ((first >= 1) & (first <= horizon)))):
            raise ValueError("Invalid per-repeat coverage records")
        reconstructed = ((first[:, None] == -1) | (first[:, None] > x)).sum(axis=0)
        if not np.array_equal(reconstructed, counts):
            raise ValueError("Per-repeat records disagree with the coverage summary")

    alpha = float(step.alpha.iloc[0])
    values = pd.DataFrame(dict(
        step=x, n_repeats=repeats, n_covered_through_t=counts,
        coverage=counts / repeats,
        coverage_mc_lower=step.coverage_mc_lower.to_numpy(),
        coverage_mc_upper=step.coverage_mc_upper.to_numpy(),
        scenario="tilde_z_2", method="bet_uniform", alpha=alpha,
        n_questions=int(step.n_questions.iloc[0]),
    ))
    stem = "validity_tilde_z2_bet_uniform_r800"
    values.to_csv(output / f"{stem}_values.csv", index=False)
    values.tail(1).to_csv(output / f"{stem}_terminal.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.6, 4))
    label, color, marker = METHODS["bet_uniform"]
    positions = np.linspace(0, len(x)-1, min(10, len(x)), dtype=int)
    ax.plot(x, values.coverage.to_numpy(), color=color, linestyle="-",
            marker=marker, markevery=positions, ms=4, lw=1.5, label=label)
    ax.fill_between(x, values.coverage_mc_lower.to_numpy(),
                    values.coverage_mc_upper.to_numpy(), color=color, alpha=.16)
    ax.axhline(1-alpha, color="black", ls=":", lw=1.2,
               label=f"Nominal {100*(1-alpha):g}% coverage")
    ax.set_title(VALIDITY_LABELS["tilde_z_2"], fontsize=16)
    ax.set_xlabel(r"Queries $t$", fontsize=13)
    ax.set_ylabel(r"Coverage at all steps through $t$", fontsize=12)
    ax.set_xlim(0, horizon)
    ymin = min(.9, float(values.coverage_mc_lower.min()) - .015)
    ax.set_ylim(max(0, ymin), 1.01)
    if ymin == .9:
        ax.set_yticks(np.arange(.9, 1.001, .02))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.tick_params(labelsize=11)
    ax.grid(alpha=.2)
    ax.legend(loc="lower left", fontsize=10)
    return save_figure(fig, output, stem)


def rate_reference(t, kappa, family):
    if np.isclose(kappa, 0):
        return np.ones_like(t), "1"
    if family == "ripr" and np.isclose(kappa, 1):
        return (1+np.log(t))/t, r"(1+\log t)/t"
    exponent = min(kappa, .5 if family == "bet" else 1.)
    label = r"t^{-1/2}" if np.isclose(exponent, .5) else rf"t^{{-{exponent:g}}}"
    return t**(-exponent), label


def plot_mismatch(step, output, anchor_step):
    oracle = step[(step.predictor == "oracle") & step.method.isin(("ripr_uniform", "bet_uniform"))]
    if oracle.scenario.nunique() != 1 or oracle.empty:
        raise ValueError("Theory figure requires one scenario with both oracle uniform methods")
    scenario = oracle.scenario.iloc[0]
    kappas = sorted(oracle.kappa.unique())
    colors = {k: plt.cm.tab10.colors[j % 10] for j, k in enumerate(kappas)}
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9.2, 3.8))
    anchors = []
    for ax, family in zip(axes, ("ripr", "bet")):
        for kappa in kappas:
            g = oracle[(oracle.method == f"{family}_uniform") & oracle.kappa.eq(kappa)].sort_values("step")
            if g.empty or not g.step.eq(anchor_step).any():
                raise ValueError(f"Missing {family}, kappa={kappa} at anchor t={anchor_step}")
            t, y = g.step.to_numpy(float), g.width_mean.to_numpy()
            se, color = g.width_se.fillna(0).to_numpy(), colors[kappa]
            reference, label = rate_reference(t, kappa, family)
            valid = np.isfinite(y) & (y > 0)
            ax.loglog(t[valid], y[valid], color=color, label=rf"$\kappa={kappa:g}$: ${label}$")
            ax.fill_between(t[valid], np.maximum(y[valid]-se[valid], 1e-12),
                            y[valid]+se[valid], color=color, alpha=.12)
            at = np.flatnonzero(valid & (t >= anchor_step))
            if not len(at):
                raise ValueError("No positive width at or after the reference anchor")
            at = at[0]
            scale = y[at]/reference[at]
            ax.loglog(t, reference*scale, color=color, ls="--", lw=1)
            anchors.append(dict(family=family, kappa=float(kappa), anchor_step=int(t[at]),
                                empirical_width=float(y[at]), scale=float(scale), rate=label))
        for method in BASELINES:
            g = step[(step.scenario == scenario) & (step.method == method)].sort_values("step")
            if g.empty or g.duplicated("step").any():
                raise ValueError(f"Missing or duplicated mismatch baseline: {method}")
            if g.z_sha256.nunique() != 1 or g.z_sha256.iloc[0] != oracle.z_sha256.iloc[0]:
                raise ValueError("Mismatch baseline uses different fixed labels")
            label, color, _ = METHODS[method]
            valid = g.width_mean > 0
            t, y = g.step[valid].to_numpy(), g.width_mean[valid].to_numpy()
            se = g.width_se[valid].fillna(0).to_numpy()
            # This qualifier appears in the supplied theory screenshot only.
            ax.loglog(t, y, color=color, label=label+" (no side info)", lw=1.4)
            ax.fill_between(t, np.maximum(y-se, 1e-12), y+se, color=color, alpha=.12)
        ax.set_xlabel("Queries $t$ (log scale)")
        ax.set_ylabel(f"Mean {'RIPr' if family == 'ripr' else 'Betting'}-CS width (log scale)")
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="lower left")
    pd.DataFrame(anchors).to_csv(output / "theory_guide_anchors.csv", index=False)
    return save_figure(fig, output, "width_vs_theory_rate_combined")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="Output root for main_figure/, appendix_figure/ and figures/ (default: package directory)")
    parser.add_argument("--only", choices=("all", *EXPERIMENTS), default="all")
    parser.add_argument("--anchor-step", type=int, default=1000)
    for experiment in ("comparison", "validity", "mismatch"):
        parser.add_argument(f"--{experiment}-results", type=Path,
                            help="Use newly simulated run(s), recursively pooling completed parts")
    parser.add_argument("--validity-r800-results", dest="validity_r800_results", type=Path,
                        help="Use a new standalone tilde_z_2 / betting-uniform validity run")
    parser.add_argument("--m2-results", dest="m2_comparison_results", type=Path, nargs="+",
                        help="One or more completed M2 run directories containing all four models")
    parser.add_argument("--bbh-m2-results", dest="bbh_m2_comparison_results", type=Path, nargs="+",
                        help="Completed combined-suite runs for rows 788, 1752, 1516 and 729")
    args = parser.parse_args()
    if args.anchor_step < 1:
        parser.error("--anchor-step must be positive")
    cases = EXPERIMENTS if args.only == "all" else (args.only,)
    if any(getattr(args, f"{case}_results") is None for case in cases):
        verify_data(args.data_dir)
    output = args.output_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "pdf.fonttype": 42})
    files, provenance = [], {}
    for case in cases:
        step, stop, sources = read_summaries(args.data_dir / case, getattr(args, f"{case}_results"))
        provenance[case] = sources
        if case == "comparison":
            for kind in ("stopping", "width", "mae", "kl"):
                files.append(plot_comparison(step, stop, output, kind))
            files.append(plot_comparison(step, stop, output, "stopping", paired=True))
        elif case == "validity":
            files.extend(plot_validity(step, output))
        elif case == "validity_r800":
            repeat_coverage = None
            if args.validity_r800_results is None:
                repeat_coverage = args.data_dir / case / "repeat_coverage.csv"
                sources[f"{case}/repeat_coverage.csv"] = file_hash(repeat_coverage)
            files.append(plot_validity_r800(step, output, repeat_coverage))
        elif case == "m2_comparison":
            files.extend(plot_m2_comparison(step, stop, output))
        elif case == "bbh_m2_comparison":
            files.extend(plot_m2_comparison(step, stop, output, scenarios=BBH_M2_SCENARIOS,
                         kinds=("stopping", "kl", "mae"), suffix="bbh_m2_comparison"))
        else:
            files.append(plot_mismatch(step, output, args.anchor_step))
    figure_files = {stem: {ext: figure_path(output, stem).with_suffix(f".{ext}").relative_to(
        args.output_dir).as_posix() for ext in ("pdf", "png")} for stem in files}
    manifest = dict(schema_version=2, figures=files, figure_files=figure_files,
                    paths_relative_to="output root", source_hashes=provenance,
                    uncertainty="+/-1 SE; pointwise 95% Clopper-Pearson Monte Carlo intervals for coverage",
                    theory="Rate guides normalized to empirical width at the recorded anchor; not numerical bounds",
                    plots_include_paper_captions=False)
    (output / f"figure_manifest_{args.only}.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Reproduced {len(files)} figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
