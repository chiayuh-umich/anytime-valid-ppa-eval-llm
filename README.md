# Anytime-Valid Prediction-Powered Active Evaluation of Large Language Models

**FAQ predictor source:** the factor-predictor initialization and sequential
Laplace updates are adapted from [Skyler Wu et al.'s efficiently-evaluating-llms
repository](https://github.com/skbwu/efficiently-evaluating-llms), specifically
[`faq_final.py`](https://github.com/skbwu/efficiently-evaluating-llms/blob/main/faq_final.py).
The upstream Apache-2.0 license is included as `LICENSE-FAQ`. This attribution
concerns the FAQ predictor; the nested CS constructions and querying rules are
implemented in this package.

This directory is a self-contained reproduction package for the supplied paper
figures. Copy this entire directory to another machine: no Python modules or
data outside it are needed to redraw the figures or rerun their experiments.
The original project files and results have not been moved or modified.

There are two workflows:

1. **Redraw the recorded results.** This takes seconds and uses the included
   summary data. It does not run any sequential experiments.
2. **Simulate new results.** This uses the included fixed labels and predictor
   snapshots. The original experiments are expensive; a GPU is recommended for
   the complete 5000-question runs. Small CPU smoke runs are also supported.

## Contents

```text
nested_reproduction/
  README.md
  nested_experiment.py          All simulation and numerical solver code
  reproduce_figures.py          Original plots, M2 comparisons and r800 validity
  requirements.txt             Plotting dependencies
  requirements-experiments.txt Simulation dependencies
  verification.json           Executed checks and source-code hashes
  LICENSE-FAQ                 Upstream license for the adapted FAQ predictor
  data/
    benchmark_5000/             Six fixed binary vectors + FAQ predictor snapshot
    m2_row_0483/                Additional real M2 model, 5000 questions
    m2_row_0542/                Additional real M2 model, 5000 questions
    m2_row_1183/                Additional real M2 model, 5000 questions
    m2_row_1506/                Additional real M2 model, 5000 questions
    comparison/                Recorded summaries: 20 repeats, 5000 steps
    validity/                  Recorded summaries: 300 repeats, 500 steps
    validity_r800/             Standalone tilde_z_2 / betting uniform: 800 repeats
    mismatch/                  Recorded summaries: 20 repeats, 5000 steps
    m2_comparison/              Four real M2 models: 20 repeats, 5000 steps
    initial_mismatch.csv       Initial means, MAE, Brier and KL
    provenance.json            Sources and retained data definitions
    checksums.json             SHA-256 hashes of all distributed data files
  main_figure/                 Paper Figures 1--2, numbered PDF and PNG files
  appendix_figure/             Paper Figures 3--10, numbered PDF and PNG files
  figures/                    Additional diagnostics, numeric tables and metadata
```

`nested_experiment.py` contains the RIPr projection, exact betting-fraction
solver, query optimization, both Hedged baselines, predictors, nested updates,
data readers, result pooling and general diagnostic plotting. It does not import
the original `fixed_z_experiment.py`, `nested_m2_experiment.py`, or any of their
helper modules. `reproduce_figures.py` imports the consolidated runner only when
asked to read newly simulated run directories.

The FAQ predictor includes both initialization and updates:

- `load_data` / `load_prepared` load `v`, `mu`, `sigma` from `predictor.npz`.
  The stored initialization was computed as `mu = mean(U, axis=0)` and
  `sigma = cov(U.T)` from the fitted historical model factors.
- `FAQPredictor.__init__` gives each independent repeat its own copies of
  `u = mu` and `Sigma = sigma`.
- `FAQPredictor.predict` computes `sigmoid(u @ v.T)`, using this runner's
  float64 arithmetic and `[1e-6, 1-1e-6] clipping.
- `FAQPredictor.update` calls `advance_predictor` and `update_factor_posterior`
  after observing the queried response. The covariance is updated first, then
  the model factor mean is updated with that covariance and the prediction
  residual. No unqueried labels are used by the FAQ predictor.

The fitted factors are already supplied; retraining the historical factor model
is not part of a nested simulation.

## Quick start: reproduce the numbered paper figures

Run these commands inside this directory. The tested environment is Python
3.10 with the versions pinned in the requirements files.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_figures.py
```

The plotting script verifies the distributed data hashes before use. The default
command redraws **10 numbered paper figures** directly into `main_figure/` and
`appendix_figure/`, using the filenames below. Each PDF also has a PNG with the
same stem. The numbering is fixed in the script: the output folders need not
already contain any figures. Running it again overwrites the matching output
files. Six additional diagnostic plots and the numeric tables go in `figures/`.
It uses a headless Matplotlib backend; LaTeX and PyTorch are not needed.

| Paper figure | PDF output relative to this directory | Content |
| --- | --- | --- |
| 1 | `main_figure/1_width_vs_theory_rate.pdf` | RIPr and betting, shared oracle predictor, 1 × 2 log–log panels |
| 2 | `main_figure/2_stopping_time_z123.pdf` | Capped stopping queries for the three unpermuted datasets |
| 3 | `appendix_figure/3_CS_width_z123.pdf` | Mean CS width for the three datasets |
| 4 | `appendix_figure/4_one_step_mae_z123.pdf` | Betting predictor MAE, three panels |
| 5 | `appendix_figure/5_one_step_kl_query_z123.pdf` | RIPr query-weighted KL, three panels |
| 6 | `appendix_figure/6_stopping_time_paired_z123.pdf` | Unpermuted and permuted datasets, 3 × 2 panels |
| 7 | `appendix_figure/7_stopping_time_m2_comparison.pdf` | Capped stopping queries versus epsilon, four models in a 2 × 2 layout |
| 8 | `appendix_figure/8_one_step_mae_m2_comparison.pdf` | One-step MAE for the four M2 models, betting methods only |
| 9 | `appendix_figure/9_validity_through_t.pdf` | Coverage through each time, four panels, 300 repeats |
| 10 | `appendix_figure/10_validity_tilde_z2_bet_uniform_r800.pdf` | Betting uniform on tilde z_2 only, 800 repeats, 500 steps |

The six additional plot stems under `figures/` are `validity_summary`,
`width_m2_comparison`, `coverage_m2_comparison`, `one_step_kl_query_m2_comparison`,
`cumulative_kl_query_m2_comparison`, and `cumulative_mae_m2_comparison`.

The screenshots' surrounding paper captions and page whitespace are not part
of the generated plots. Colors, method ordering, markers and subplot layouts
follow the supplied figures. Only the theory figure retains the screenshot's
`(no side info)` qualifier in the two baseline labels.

To draw only one experiment group into the same numbered folders:

```bash
python reproduce_figures.py --only comparison
python reproduce_figures.py --only validity
python reproduce_figures.py --only mismatch
python reproduce_figures.py --only m2_comparison
python reproduce_figures.py --only validity_r800
```

`--output-dir` now specifies an **output root**, rather than a flat figure
folder. For example, `python reproduce_figures.py --output-dir reproduced`
creates `reproduced/main_figure/`, `reproduced/appendix_figure/`, and
`reproduced/figures/`. Omitting this option uses the package directory,
regardless of the current working directory.

Under `figures/`, `validity_terminal_values.csv` reports the plotted coverage
and Monte Carlo intervals, and `theory_guide_anchors.csv` records the exact
scaling of each dashed reference curve. Each invocation also writes
`figure_manifest_<group>.json`, whose `figure_files` field lists the actual
PDF/PNG paths relative to the output root.

### Standalone validity figure: 800 repeats

To redraw only the additional betting-uniform validity figure:

```bash
python reproduce_figures.py --only validity_r800
```

This writes `appendix_figure/10_validity_tilde_z2_bet_uniform_r800.pdf` and
`.png`, plus `figures/validity_tilde_z2_bet_uniform_r800_values.csv` (all 500
plotted points) and `figures/validity_tilde_z2_bet_uniform_r800_terminal.csv`
(the final point). It does
not combine this run with other methods or replace the four-panel r300 figure.

The included run uses the FAQ predictor, uniform sampling with replacement,
the fixed `tilde_z_2` vector of 5000 questions, alpha 0.05, and 800 repeats
(IDs 0--799, seed 0), each followed for 500 queries. At step 500, 763 of 800
repeats retain the true candidate: coverage is **95.375%**, with a pointwise
95% Clopper--Pearson interval of **[93.681%, 96.723%]**. The solid line shows
the fraction covered at every step through t; shading shows these pointwise
Monte Carlo intervals, and the dotted line marks nominal 95% coverage.

`data/validity_r800/` contains:

- `step_summary.csv.gz` and `stopping_summary.csv.gz`: the original recorded
  summary tables, compressed without changing their numeric values.
- `repeat_coverage.csv`: one row per repeat, with its `repeat_id`, `horizon`,
  and `first_exclusion_step`. A value of -1 means the true candidate survived
  the complete recorded horizon; it does not mean survival beyond step 500.
  At time t, a repeat counts as covered exactly when this value is -1 or
  greater than t. Because the CS is nested, this compact table reconstructs
  every repeat's coverage trajectory. The plotting script verifies that its
  counts match the summary at all 500 times before drawing the figure.
- `source_runs.json`: portable experiment settings, without machine-specific
  paths. The labels and FAQ initialization are in `data/benchmark_5000/`.

The first-exclusion records were checked against all 400,000 original
per-repeat trajectory rows. Other raw trajectory fields are not included in
this compact addition. The first 300 repeat IDs reuse the original validity
experiment's random streams; **do not pool the 300 and 800 runs as 1100
independent repeats**. Instructions to simulate a new run are given below.

## Included datasets

All paper experiments use **N = 5000** binary answers. Each scenario has one
fixed label vector, not an average across multiple models. In a repeat, randomness
comes from querying; the ground truth remains fixed and the predictor state is
reset. The target is the finite-population mean `theta_star = sum(z) / 5000`.

| Paper label | Scenario | Construction | Mean |
| --- | --- | --- | --- |
| z_1 | `z_1` | Unmodified M2 row 1, first 5000 stored questions | 0.3956 |
| z_2 | `z_2` | Set the 3500 largest initial prediction scores to one | 0.7000 |
| z_3 | `z_3` | Set the 4500 largest initial prediction scores to one | 0.9000 |
| tilde z_1 | `tilde_z_1` | Fixed permutation of z_1 | 0.3956 |
| tilde z_2 | `tilde_z_2` | Fixed permutation of z_2 | 0.7000 |
| tilde z_3 | `tilde_z_3` | Fixed permutation of z_3 | 0.9000 |

The real model in z_1 is `bamec66557__MISCHIEVOUS-12B-Mix_0.6v`.
All row numbers are **zero-based**. The question columns are the first 5000 in
the stored MMLU-Pro M2 matrix; no new question sampling is done during preparation.
The tilde variants are synthetic controls, including `tilde_z_1`.

Use ASCII names `z_1`, `z_2`, `z_3`, `tilde_z_1`, `tilde_z_2`, `tilde_z_3`
in commands and CSV files. Plots display subscripted mathematical labels
`$z_1$` through `$z_3$` and `$\tilde z_1$` through `$\tilde z_3$`.
The six label files are named `z_1.csv`, ..., `tilde_z_3.csv`.

For tracing earlier results, the naming correspondence is:

| Previous name | Current name | Preserved random-stream ID |
| --- | --- | --- |
| `small_similar` | `z_1` | 4 |
| `small_medium` | `z_2` | 2 |
| `small_high` | `z_3` | 0 |
| `large_similar` | `tilde_z_1` | 5 |
| `large_medium` | `tilde_z_2` | 3 |
| `large_high` | `tilde_z_3` | 1 |

The runner accepts the old names when reading earlier datasets/runs and converts
them to the current names. Historical source paths/names remain in provenance
records. Renaming does not change the labels, random draws or numerical curves.

The initial FAQ predictor has mean approximately 0.318333. `predictor.npz`
contains the 5000 × 64 question factors `v`, 64-dimensional factor mean `mu`,
and 64 × 64 covariance `sigma`. These are the already fitted initialization:
the package does not need the original training matrix or retrain the factor
model. Each data manifest records construction, model identity and checksums.
The large original M2 file is not distributed; the selected binary answers are.

Additional previously prepared model banks are included for the same runner:

| Directory / scenario | Mean | Model |
| --- | --- | --- |
| `data/m2_row_0483` / `m2_row_0483` | 0.3694 | See its manifest |
| `data/m2_row_1183` / `m2_row_1183` | 0.1016 | See its manifest |
| `data/m2_row_1506` / `m2_row_1506` | 0.2500 | `mobiuslabsgmbh__DeepSeek-R1-ReDistill-Llama3-8B-v1.1` |
| `data/m2_row_0542` / `m2_row_0542` | 0.3798 | `hotmailuser__QwenSlerp-3B` |

These four models contribute to the seven additional M2 comparison figures.
They are distinct from the six z/tilde-z datasets in the original eight figures.
The new panels are ordered by increasing ground-truth accuracy: row 1183
(10.16%), row 1506 (25.00%), row 483 (36.94%), and row 542 (37.98%). Each
panel contains all eight methods; models are not pooled into a single average.
All four use the same first 5000 question columns and identical initial FAQ
predictor snapshots. `data/m2_comparison/models.csv` reports their identities,
initial mismatch, input hashes and source groups. The model names identify
external benchmark models, not the authors of this reproduction package.

The recorded results have 20 repeats per model/method, horizon 5000,
alpha 0.05 and epsilon values 0.05, 0.075, 0.1, 0.125, 0.15. The stopping
figure shows mean capped stopping queries with one-standard-error shading;
a hollow marker means some repeats did not reach the target by the horizon.
The coverage figure uses pointwise 95% Clopper--Pearson Monte Carlo intervals,
not simultaneous bands. The plotting script also exports the exact stopping
values and terminal coverage table to the output root's `figures/` directory.

Four further 2 × 2 figures report one-step and cumulative prediction mismatch.
Only RIPr methods appear in the KL figures, which use
`sum_i q_t(i) * KL(Bernoulli(z_i) || Bernoulli(r_(t-1)(i)))`.
Only betting methods appear in the MAE figures, which use
`mean_i abs(r_(t-1)(i)-z_i)` without query weighting. Cumulative mismatch sums
these pre-query losses from step 1 through t. Shading is one standard error
across repeats; the cumulative standard errors come from the per-repeat
cumulative losses, not sums of one-step standard errors. All curves use every
recorded step and are shown on linear axes. Hedged baselines have no
per-question predictor and are omitted from the mismatch figures.

## Rerun the experiments

```bash
python -m pip install -r requirements-experiments.txt
```

The commands below reproduce the original experimental settings. Choose
`--device cuda` on a CUDA GPU or `--device cpu` for a CPU. No Slurm settings,
account names or environment-module paths are hard-coded. An output directory
must be absent or empty: the runner refuses to overwrite existing results.

### Validity: four datasets, six methods, 300 repeats

```bash
python nested_experiment.py run \
  --data-dir data/benchmark_5000 --output-dir results/validity \
  --scenarios z_2 z_3 tilde_z_2 tilde_z_3 \
  --methods ripr_uniform ripr_maxmin bet_uniform bet_maxmin hedged_uniform hedged_wor \
  --predictor faq --n-repeats 300 --max-steps 500 \
  --seed 0 --alpha 0.05 --uniform-weight 0.05 --grow-opt-steps 25 \
  --repeat-batch-size 8 --candidate-batch-size 256 --threads 2 \
  --device cuda --no-plots
python reproduce_figures.py --only validity \
  --validity-results results/validity --output-dir reproduced/rerun_validity
```

The population still contains 5000 questions; only the simulation horizon is
500. Max-max querying was not included in this recorded validity experiment.

### Validity follow-up: tilde z_2, betting uniform, 800 repeats

The following command reruns the standalone experiment using only packaged
inputs. It is much slower than redrawing the included results.

```bash
python nested_experiment.py run \
  --data-dir data/benchmark_5000 --output-dir results/validity_r800 \
  --scenarios tilde_z_2 --methods bet_uniform --predictor faq \
  --n-repeats 800 --repeat-start 0 --max-steps 500 \
  --seed 0 --alpha 0.05 --uniform-weight 0.05 \
  --epsilons 0.05 0.075 0.1 0.125 0.15 \
  --repeat-batch-size 8 --candidate-batch-size 256 --threads 2 \
  --device cuda --no-plots
python reproduce_figures.py --only validity_r800 \
  --validity-r800-results results/validity_r800 \
  --output-dir reproduced/rerun_validity_r800
```

Use `--device cpu` if CUDA is unavailable. The runner still records complete
trajectories with `--no-plots`; this flag only skips its generic diagnostics.
The archived run used CUDA. Numerical results from fresh simulations may
differ slightly across devices or library versions; redrawing the bundled
recorded tables reproduces the supplied numerical coverage values exactly.
The existing simulation code and predictor updates are unchanged.

### Controlled mismatch: shared predictor, three decay powers

```bash
python nested_experiment.py run \
  --data-dir data/benchmark_5000 --output-dir results/mismatch \
  --scenarios z_1 \
  --methods ripr_uniform bet_uniform hedged_uniform hedged_wor \
  --predictor oracle --kappas 0 0.5 1 --oracle-amplitude 0.9 --oracle-center 0.7 \
  --n-repeats 20 --max-steps 5000 --seed 0 --alpha 0.05 \
  --repeat-batch-size 8 --candidate-batch-size 256 --threads 2 \
  --device cuda --no-plots
python reproduce_figures.py --only mismatch \
  --mismatch-results results/mismatch --output-dir reproduced/rerun_mismatch
```

Before query s, both CS families use the same oracle predictor

```text
r_(s-1)(i) = (1 - a_s) z_i + a_s c,
a_s = 0.9 s^(-kappa),   c = 0.7,   kappa in {0, 0.5, 1}.
```

Thus all three powers have the same initialization. Oracle access to the whole
label vector is intentional only in this controlled experiment. It is not given
to the FAQ predictor or the Hedged methods. Each Hedged baseline is simulated
once per scenario and displayed in both panels, not counted as extra repeats.

### Comparison: six datasets, eight methods, 20 repeats

```bash
python nested_experiment.py run \
  --data-dir data/benchmark_5000 --output-dir results/comparison \
  --scenarios z_1 z_2 z_3 tilde_z_1 tilde_z_2 tilde_z_3 \
  --methods ripr_uniform ripr_maxmin ripr_maxmax bet_uniform bet_maxmin bet_maxmax hedged_uniform hedged_wor \
  --predictor faq --n-repeats 20 --max-steps 5000 \
  --epsilons 0.05 0.075 0.1 0.125 0.15 --seed 0 --alpha 0.05 \
  --uniform-weight 0.05 --grow-opt-steps 25 \
  --repeat-batch-size 8 --candidate-batch-size 256 --threads 2 \
  --device cuda --no-plots
python reproduce_figures.py --only comparison \
  --comparison-results results/comparison --output-dir reproduced/rerun_comparison
```

For an additional real model, replace `--data-dir` by, for example,
`data/m2_row_0542` and `--scenarios` by `m2_row_0542`.

### Comparison: four real M2 models, eight methods, 20 repeats

The bundled `data/m2_row_*/` directories already contain both the binary
ground truths and the predictor snapshots, including the new mean-pair rows
1506 and 542. No download or preparation step is needed. To rerun all four
models with the settings of the recorded comparison:

```bash
for row in 1183 1506 0483 0542; do
  python nested_experiment.py run \
    --data-dir "data/m2_row_${row}" \
    --output-dir "results/m2_comparison/m2_row_${row}" \
    --scenarios "m2_row_${row}" \
    --methods ripr_uniform ripr_maxmin ripr_maxmax \
              bet_uniform bet_maxmin bet_maxmax hedged_uniform hedged_wor \
    --predictor faq --n-repeats 20 --repeat-start 0 --max-steps 5000 \
    --epsilons 0.05 0.075 0.1 0.125 0.15 --seed 0 --alpha 0.05 \
    --uniform-weight 0.05 --grow-opt-steps 25 \
    --repeat-batch-size 8 --candidate-batch-size 256 --threads 2 \
    --device cuda --no-plots
done
python reproduce_figures.py --only m2_comparison \
  --m2-results results/m2_comparison --output-dir reproduced/rerun_m2
```

Alternatively, run one method per output directory using `--methods METHOD`.
Both `--m2-results` and the export command accept multiple directories, e.g.
`--m2-results results/mean_pair results/row_1183 results/row_483`.
They pool disjoint runs and reject duplicate repeats or incomplete runs.
To regenerate the compact tables shipped under `data/m2_comparison/`:

```bash
python nested_experiment.py export \
  --results-dir results/m2_comparison \
  --experiment m2_comparison --output-dir data_rebuilt
python reproduce_figures.py --only m2_comparison \
  --data-dir data_rebuilt --output-dir reproduced/rebuilt_m2
```

The model metadata required for the plot is retained in the exported tables.
`models.csv` is an additional human-readable provenance table; it is not
required by the plotting command. New simulations are expensive; simply
redrawing the included summaries does not run any experiments.

### Run separate methods or repeat batches

Use a distinct output directory for every task. For example:

```bash
python nested_experiment.py run \
  --data-dir data/benchmark_5000 \
  --output-dir results/comparison_parts/z_1_ripr_maxmax \
  --scenarios z_1 --methods ripr_maxmax \
  --predictor faq --n-repeats 20 --max-steps 5000 \
  --threads 2 --device cuda --no-plots
```

After all desired tasks finish, use `--comparison-results results/comparison_parts`
in the paper plotting command. It recursively pools completed tasks. For the
same scenario/method, split repeats using disjoint `--repeat-start` ranges and
the same `--seed`; overlapping repeat IDs are rejected. Scenario streams are
stable when only a subset of methods/scenarios is run.

`--repeat-batch-size 8` processes at most eight independent trials at a time;
it does not change the number of trials. `--candidate-batch-size 256` limits
temporary evidence-solver tensors. `--no-plots` suppresses immediate generic
plots while still saving all trajectories and summaries.

For general plots of any run or model bank:

```bash
python nested_experiment.py plot \
  --results-dir results/comparison_parts --output-dir figures/diagnostics
```

This general command can plot a subset of scenarios/methods. The paper-specific
commands require the datasets belonging to their panels.

For a quick CPU execution check, run just two repeats for three queries:

```bash
python nested_experiment.py run \
  --data-dir data/benchmark_5000 --output-dir results/smoke \
  --scenarios z_1 --methods bet_uniform hedged_wor \
  --n-repeats 2 --max-steps 3 --device cpu --no-plots
```

## Numerical implementation

- The candidate grid is `{0, 1/N, ..., 1}`. Each trial starts with wealth one
  at every candidate. A candidate is permanently removed once its accumulated
  log wealth reaches `log(1/alpha)`. Only surviving trial/candidate pairs are
  processed in later evidence calculations. There is no RIPr or betting
  quadratic approximation in this runner.
- RIPr uses a scalar Lagrange multiplier for the mean-constrained Bernoulli
  projection. It brackets the root with at most 24 expansion steps and uses
  28 bisection steps. Null means 0 and 1 are handled exactly for evidence.
- Betting maximizes predicted expected log growth in the implemented valid
  interval `[-1/(1-m+1/(N*h)), 1/(m+1/(N*h))]`. Derivative signs select an
  endpoint optimum; otherwise 40 bisection steps solve the derivative root.
- Max-min evaluates both current nested-CS endpoints. The query distribution
  is parameterized as `q = (1-eta)*s + eta/N`, with `eta=0.05`, throughout
  optimization. The default mirror search uses at most 25 iterations per
  initialization, starts from uniform and normalized `sqrt(r*(1-r))`, and adds
  the preceding query distribution when available. Its two-endpoint mirror
  step uses 24 bisections; backtracking allows 12 trials. Defaults are step
  size 0.5, relative tolerance 1e-4 and absolute tolerance 1e-9. These are
  numerical searches, not certificates of a global optimum.
- Max-max runs the same constrained search separately for each endpoint,
  re-evaluates both proposed distributions at both endpoints, and chooses the
  larger maximum predicted growth. Both adaptive policies enforce `q_i >= eta/N`;
  uniform uses `q_i=h=1/N`.
- The FAQ predictor is `sigmoid(u @ v.T)` with a sequential Laplace factor
  posterior update after observing each queried response. Predictions are
  clipped to `[1e-6, 1-1e-6]`. All calculations use float64. The oracle schedule
  is rejected if clipping would alter its requested decay over the horizon.
- Hedged-CS and Hedged-WoR use the predictable plug-in betting strategy with
  cap 0.5 and hedge weight 0.5. They have no per-question predictor. The two
  directional capitals are tracked separately and combined by a weighted max.
  Hedged-WoR replaces each null mean by `(N*m - past_label_sum)/(N-t+1)` and
  intersects with the finite-population feasibility bounds.
- RIPr, betting and Hedged-CS query **with replacement**. Hedged-WoR queries
  uniformly **without replacement**. The x-axis counts draws; for WR methods,
  draws can revisit an already observed question. At census the WoR width is
  zero, but a previously eliminated true mean is never reinstated.

## What the plotted numbers mean

**Width:** the largest surviving candidate minus the smallest. This is the
diameter of the discrete set, not its cardinality. An empty set is recorded with
width zero, and its coverage is false.

**Capped stopping queries:** the first time a *nonempty* nested CS has full
width at most epsilon, capped at the simulation horizon H. A hollow marker
indicates that at least one trial did not reach this target by H. This is not
an estimate of the uncensored expected stopping time. Simulations continue to
H even after a target is first reached so complete trajectories are available.

**Coverage through t:** the fraction of independent trials whose true candidate
survives at every step through t. Because the sets are nested, it equals current
exact-set membership. The dotted horizontal line is 95%. Error bars and shaded
coverage regions are pointwise two-sided 95% Clopper–Pearson Monte Carlo
intervals, not a simultaneous confidence band over times/methods. Other shaded
regions are mean ± one standard error across repeats. Observed coverage below
95% is retained as recorded, not adjusted.

**One-step mismatch:** at query t, use the pre-query prediction `r_(t-1)`.
The RIPr plot reports
`sum_i q_t(i) * [-z_i log(r_i) - (1-z_i) log(1-r_i)]`.
The betting plot reports MAE, `mean_i abs(r_i-z_i)`, **not** query-weighted MAE
or Brier error. Natural logarithms are used for KL. No temporal smoothing is
applied. These plots average predictor losses across repeats.

**Dashed theory curves:** these are leading-order reference shapes, normalized
to each empirical curve at t=1000. They are **not numerical upper bounds** or
independently predicted widths. Fixed-N logarithmic constants, including
`log N` and `log(1/alpha)`, are not evaluated as bound constants in these guides.

| Shared predictor power kappa | RIPr guide | Betting guide |
| --- | --- | --- |
| 0 | 1 | 1 |
| 0.5 | t^(-1/2) | t^(-1/2) |
| 1 | (1+log t)/t | t^(-1/2) |

For the shared oracle, one-step KL and MAE are of order `a_s`, whereas Brier
is of order `a_s^2`. The current betting reference keeps the baseline
`sqrt(t)/t` term, which explains its t^(-1/2) guide even at kappa=1. This plot
compares the existing experiments with the stated theoretical rate shapes;
it does not by itself establish a decay exponent or show a bound is tight.
Also, the existing betting proof's restricted/half-range step and the code's
full implemented betting interval differ. Consolidation preserves that interval;
the dashed line should not be described as a verified finite-sample bound for
this implementation. Zero widths are omitted on logarithmic axes.

## Outputs and provenance

Each new simulation writes:

- `config.json`: complete settings, data/code hashes and completion status;
- `*_steps.csv`: per-repeat trajectories, queried indices/labels, bounds,
  exact coverage, active candidate counts and mismatch diagnostics;
- `*_stops.csv`: per-repeat stopping/censoring and coverage values;
- `*_eliminations_*.npz`: the elimination time of each candidate (zero if still active);
- `step_summary.csv`, `stopping_summary.csv`: means and standard errors;
- `*_timing.csv`: elapsed time and candidate-evaluation counts by repeat batch.

The distributed `data/{comparison,validity,mismatch,m2_comparison}` folders contain compact
pooled summaries sufficient to reproduce the figures. They retain every plotted
time step and threshold, not downsampled curves. They do not include the much
larger original per-repeat trajectory files, so they cannot support a new
per-trial statistic or an arbitrary new stopping threshold without rerunning.
`source_runs.json` records the original completed-run settings and hashes;
`provenance.json` maps the data to the source experiment directories. Machine-
specific home paths have been converted to project-relative provenance paths.

### How to generate the files under data/

The directory contains **inputs** as well as **recorded outputs**:

| Files | Role | How they are obtained |
| --- | --- | --- |
| `benchmark_5000/*.csv` | Fixed ground-truth labels | The prepared M2 row and synthetic constructions listed above; supplied as simulation inputs |
| `benchmark_5000/predictor.npz` | Initial FAQ parameters | Saved fitted question factors and historical factor mean/covariance; supplied as input |
| `m2_row_*/` | Other prepared models | The `prepare-m2` command, using an external M2 CSV and a predictor snapshot |
| `{comparison,validity,mismatch,m2_comparison}/*_summary.csv.gz` | Compact plotting results | `run` generates trajectories and summaries; `export` pools completed runs and creates these compressed tables |
| `{comparison,validity,mismatch,m2_comparison}/source_runs.json` | Result provenance | Written by `export` from the completed run configurations |
| `validity_r800/` | Standalone 800-repeat coverage results | Original summaries and compact first-exclusion records, as described in the validity follow-up section |
| `m2_comparison/models.csv` | Four-model overview | Collected from the bundled input manifests and checked against the recorded runs |
| `checksums.json` | Data integrity | Written/updated by `export` for the exported data directory |

The `run` commands above generate the results from the supplied input labels
and predictor parameters. They do not generate or train those inputs. To rebuild
the same **plotting-data format** after completing the experiment commands,
run:

```bash
python nested_experiment.py export \
  --results-dir results/comparison --experiment comparison --output-dir data_rebuilt
python nested_experiment.py export \
  --results-dir results/validity --experiment validity --output-dir data_rebuilt
python nested_experiment.py export \
  --results-dir results/mismatch --experiment mismatch --output-dir data_rebuilt
python nested_experiment.py export \
  --results-dir results/m2_comparison --experiment m2_comparison --output-dir data_rebuilt
for group in comparison validity mismatch m2_comparison; do
  python reproduce_figures.py --only "$group" \
    --data-dir data_rebuilt --output-dir reproduced/rebuilt
done
```

Each export accepts either a single completed run or a parent containing disjoint
completed tasks. It preserves every time step, the per-step means/SEs, stopping
summaries, and original-run provenance. Repeating an export to a nonempty
experiment subdirectory is rejected. A new `data_rebuilt` directory containing
only these four result groups is sufficient to redraw their fifteen figures,
using the four explicit `--only` commands above. For the additional Figure 10,
use the bundled `--only validity_r800` command or the documented
`--validity-r800-results results/validity_r800` command after its new simulation.
The input model snapshots are needed only when running simulations.

The end-to-end path is therefore:

```text
data/benchmark_5000 (fixed labels + predictor initialization)
    -> nested_experiment.py run    -> results/<experiment>/
    -> nested_experiment.py export -> data_rebuilt/<experiment>/
    -> reproduce_figures.py        -> figures/rebuilt/
```

The package preserves the original seeds and scenario stream numbering.
Random streams use `SeedSequence([seed, scenario_index, repeat_id])` and are
paired across methods and oracle powers. New source-file hashes necessarily
differ after consolidation. Full GPU reruns can also differ slightly across
PyTorch/device versions, particularly near optimizer ties and rejection
thresholds. Redrawing the recorded summaries reproduces the recorded numerical
curves directly; raster text rendering can depend on fonts/platform.

`verification.json` records the checks performed during consolidation: exact
agreement with the previous runner for all eight methods and six oracle cases
on a small CPU problem; standalone CLI runs for FAQ, oracle and custom
predictors; data preparation and diagnostic plotting; and all eight figures
redrawn in an isolated copy with PyTorch imports explicitly blocked. The full
large experiments were not rerun as part of packaging.

The M2 extension additionally checks all 32 completed model/method runs and
160,000 exported step rows, and recomputes all 160 stopping-summary points
from the original per-repeat stopping records. All fifteen plots were redrawn
in an isolated copy using only bundled data, with both PyTorch and the
simulation runner imports blocked. Multi-directory plotting was also checked
against the compact export, and duplicate source repeats are rejected.
The cumulative KL and MAE curves were also checked against the running sums
of the corresponding one-step means for all model/method combinations.

## Custom predictors and additional M2 rows

A custom predictor factory receives one context dictionary with `v`, `mu`,
`sigma`, `n_questions`, `n_repeats`, `repeat_ids`, `device`, `seed`, `scenario`,
`max_steps`, and JSON `parameters`. It returns an object with:

```python
def predict(self, step):
    # Before the current response: return shape (N,) or (n_repeats, N).
    # Every entry must be in [1e-6, 1-1e-6].
    ...

def update(self, step, indices, labels):
    # Optional; called after evidence is updated, with only queried responses.
    ...
```

Use `--predictor custom --predictor-factory /path/model.py:make_predictor`
and optionally `--predictor-kwargs '{"parameter": 1}'`. The context contains
`z=None` unless `--custom-oracle` is explicitly supplied. Custom predictors must
use only information available before the current query for predictable bets.

Existing prepared M2 banks are supported directly. If the original M2 CSV is
available separately, select new rows with the bundled predictor snapshot:

```bash
python nested_experiment.py prepare-m2 \
  --m2-path /path/to/M2.csv --model-rows 1506 542 \
  --reference-row 1 --n-questions 5000 --output-dir data/new_m2_bank
```

Omit `--model-rows` to select the model with the largest absolute mean difference
from the reference, with ties resolved by row order. For a different benchmark,
provide its correctly aligned `--predictor-snapshot` containing `v`, `mu`,
`sigma`; question-factor alignment must be maintained. No factor-model training
is performed by this preparation command.
