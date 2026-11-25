# Prediction, selection, combination and evaluation

`run_combine.py` and `eforecast/combine_predictions/ensemble.py` replace
`run_combine_models.py`, `prediction/predict.py`, `prediction/evaluate.py`,
`combine_predictions_fit.py`, the CatBoost combine classifier and its trainer.  The step comes
after `run_training.py` (GPU) and `run_training.py --cpu`.

## Protocol

The evaluation period (after `evaluation_start`) is split into **three** slices in time order.
`selection_fraction` (methods section, default 0.5) marks off the part that may influence anything;
the last `calibration_fraction` of that part (default 0.3) becomes the **calibration** slice and the
rest is the **selection** slice; everything after it is the **holdout**.

* **selection** chooses the trials and fits the combiners;
* **calibration** fits the conformal bands and nothing else;
* **holdout** is seen only by the final report, so its numbers are an honest estimate.

The calibration slice comes out of the selection part, not out of the holdout, so the reported
numbers keep the sample they had. `calibration_fraction: 0` puts the bands back on the selection
slice — what the pipeline did before — and logs why that under-covers.

Each boundary is purged of the target-window overlap, as the training partition is: a sample whose
target reaches into the next slice is dropped from the earlier one. Before these changes the whole
evaluation period served the checkpoint choice, the trial ranking and the combiner evaluation at
once, and then the bands were fitted on the dates that had chosen the predictor.

## Stages

```bash
python run_combine.py --config <case>                    # predict, select, combine, evaluate
python run_combine.py --config <case> --stage select
python run_combine.py --config <case> --refit            # recompute cached predictions, refit combiners
```

| stage | what it does | output |
|---|---|---|
| predict | every trial of every method (global and clusters, GPU and CPU) predicts the evaluation period once | `<trial>/predictions_eval.pickle` (cache) |
| select | per (cluster, method) the `keep_best_models` trials with the lowest MAE on the selection slice. A unit with **no selection support** — no rows of its cluster in the selection slice, so its completed trials cannot be ranked — records that reason instead of an unexplained empty list, warns, and is listed in `Results/unsupported_clusters.csv` | `<cluster_dir>/<method>/selected_trials.json`, `Results/selection.csv`, `Results/unsupported_clusters.csv` |
| combine | kept trials of a method averaged into one predictor per method; per cluster and per horizon step the combiners of `combine_methods` are fitted on the selection slice (`kmeans` = k-means regimes on the row features with a nearest-neighbour choice of the best predictor, `bcp`, `elastic_net`; the plain `mean` is always included as the baseline); cluster averages across clusters; the "models" combiner over the global methods and the cluster averages | `<cluster_dir>/combine/<method>/hour_ahead_<h>/`, `<path_model>/combine_models/<method>/hour_ahead_<h>/` |
| evaluate | MAE, RMSE and bias in percent of the rated power (or of the target), per step and overall, on both slices, for every kept trial, method, combiner, cluster average, global method and models combiner; plus the **coverage** of each predictor against the requested timestamps of every slice; and [what the hyper-parameters of the trials say](#what-the-hyper-parameters-say-written-with-every-evaluation) | `Results/evaluation.csv` (long format), `Results/summary.csv` (holdout, all steps), `Results/coverage.csv`, `Results/hyper_parameters.csv`, `Results/hyper_parameter_report.txt` |

Levels of the result dictionary (`Ensemble.run`): `global[method]`,
`clusters[cluster][method | combiner]`, `cluster_average[name]`, `models[combiner]`.  Rows of a
cluster cover only the dates the cluster is active on; compare clusters through the cluster
averages and the models level.

## Online

```bash
python -m eforecast_online forecast --config <case> --dates 2024-04-20 2024-04-21
```

builds the online datasets for the dates, runs the kept trials and the fitted combiners only,
and returns the chosen level in power units (`predict(dates, config, path)` from code; the
project-package form of `config` still works).  `eforecast_online/forecast.py` (the root `run_model_script.py` is a wrapper of it)
is the same script.

## Notes

* Trial folders are never moved or deleted by the selection; `selected_trials.json` says which
  ones count.  Re-running `select` after new training rewrites it.
* Models saved before 2026-09-09 by the trial runner carry its monitor object pickled from
  `__main__`; `DeepNetwork.load` reads them through a tolerant unpickler, and new saves exclude
  runtime objects (monitor, data feeder, file manager), which also makes the pickles small.
* On the wind scratch case the k-means combiner beat every predictor on the selection slice and
  lost to the plain mean of the lasso trials on the holdout: the regime classifier overfits small
  selection slices.  The `mean` baseline in the report is there to catch exactly that; prefer
  `bcp` or `elastic_net` when the selection slice is short.

## Optional maintainer maintenance: slimming the training folder

For routine serving, [capture, replay and approve a bundle](user.md)
first; a finished trial is not an approval. Slimming is not required for serving. Do not slim
an approved immutable snapshot or a tree with active/unknown writers. Preview and retain a
verified backup before any explicit maintenance.

`clean_model.py` uses the selection of this stage: every `test_k` that `selected_trials.json` does
not list is moved to `<path_model>/TEMP` (same relative place), the `debug_*` folders too, and with
`--drop-checkpoints` the training checkpoints of the kept trials and of the fuzzy clusterer (a 64 MB
file per epoch in the RBF net folder).  The dataset files of `DATA` are replaced by their last
`--keep-days` days (default 30) so the online predictions can be validated against the training data;
the full files go to `TEMP/DATA` and the scaler fingerprint is rewritten so the fitted scalers stay
current.  `--dry-run` prints the plan with sizes, `--delete` removes instead of parking, `--restore`
moves everything in TEMP back file by file, `--purge-temp` deletes every TEMP folder of the model once the
slim model is validated.  Methods without a selection file are left untouched.

## Coverage: which timestamps a predictor actually answers

`evaluation.csv`'s `n` counts finite target cells, which is a different question from how many of
the requested issue times a predictor answers: nine of ten issue times with sixteen horizons each
looks fuller than it is. `Results/coverage.csv` answers the second question — requested, covered,
missing and the ratio, per predictor, level and slice — and the `evaluate` log names the predictors
that do not answer every requested holdout timestamp.

Two things make a gap. A combiner covers only the rows **every** candidate covers
(`predictors_table` takes complete cases), and a cluster average is missing wherever no *selected*
cluster covers the row — including every row that belonged only to a cluster with no selection
support. `methods.cluster_coverage_policy` decides what happens to those rows:

* `fallback_global` (the default) publishes `<name>_global_fallback` beside the combiner, completing
  the missing timestamps from the average of the global models. It is a **separate predictor**: a
  number labelled `kmeans` is kmeans wherever it is not empty, and the report shows both series
  with their coverage so the substitution is never invisible.
* `abstain` leaves the gaps, and `coverage.csv` reports them.

## What the hyper-parameters say (written with every evaluation)

Every time the `evaluate` stage runs it also answers "which hyper-parameter values did best", for
every method of the case, in two files of `Results/`:

| file | what it is |
|---|---|
| `hyper_parameters.csv` | one row per **finished trial** (kept or not, complete or pruned): the hyper-parameters that were searched, how its training ended (`state`, the `objective` the study minimised, epochs, minutes), and the MAE of its cached prediction on the **selection** and on the **holdout** dates, with its place in its study |
| `hyper_parameter_report.txt` | the same trials grouped by each hyper-parameter, method by method: the best value of each first, then a table per hyper-parameter, the best trial of each study, and whether the training's objective tells the error on unseen dates |

How it is to be read:

* **The selection dates judge, the holdout dates check.** The selection slice is the one that may
  choose, so the best value of a hyper-parameter is the one that did best there. The holdout
  column beside it only says whether that holds (`agrees`, or `differs: <value>`). Picking a value
  *because* of its holdout number turns the holdout into one more selection slice, as picking a
  predictor that way does.
* **Trials are compared inside their own study.** Clusters are not equally hard, so a trial is
  given its `place` in its study (cluster × method) — 0 for its best trial, 100 for its worst —
  and `above best %`, how far its error is above that best one. A value's numbers are the means
  over its trials in every study.
* **`evidence` is `clear`, `some` or `weak`**: how far the best value's places stand from the
  others' against the scatter that so few trials have by themselves. Weak means *do not act on
  it*; a study of fifteen trials rarely gives more for one hyper-parameter.
* **A hyper-parameter of a method is a parameter with more than one value inside one of its
  studies.** What every trial shares was not searched and is listed once. A number with many
  values is read in thirds of its range and by its rank correlation with the place.
* **The learning rate of an optimizer that finds its own step says nothing.** `prodigy` is built
  with a fixed rate (`optimizers.py`); the learning rate drawn for such a trial only sets the
  floor of its schedule, so those trials are left out of the learning-rate table, and it says so.
* **"Does the training tell the error on unseen dates?"** sets the objective of the studies
  against both slices, inside each study, and shows where the pruned and the complete trials
  stand: an objective that orders the trials differently from the unseen dates is choosing by
  something else.

Sources: `<trial>/parameters.pickle` for the networks, the study's results table
(`results_<cluster>_<method>.csv`) for the CPU methods, `trial_result.json` and `progress.json`.
A trial whose cached prediction is not of the current model and inputs — the `predict` stage of
only the kept trials leaves the others as they were — keeps its row and has no error; the report
counts them and names the command that predicts every trial.

A case evaluated before 23 November 2026 has no report, and a stage that is reused writes nothing.
So when the evaluation is asked for and its report is missing, it is made again by itself:

```bash
python run_combine.py --config <case> --stage evaluate     # or any run that includes evaluate
```

Nothing is predicted, selected or combined again, no receipt changes, and the evaluation's own
tables come out as they were; the two files are added. (`--refit` is not the way: it also does
the prediction and the combination again.)

Nothing in it can fail an evaluation or a run: a report that cannot be written is said in the
log, and the tables above are complete without it. Implementation:
`combine_predictions/hyper_parameters.py`; `run_combine.report_of_an_earlier_evaluation`.

## Conformal intervals (stage `calibrate`, 2026-09-11)

`run_combine.py --stage calibrate` (part of `all`, between `combine` and `evaluate`) fits prediction
bands on the **calibration** slice and reports them on all three, per horizon step and per coverage
level of `methods.interval_coverage` (default 0.8 and 0.9):

* **split conformal** for every predictor of the ensemble (kept trials, method averages, combiners,
  cluster averages, the models level): the residual `|y - yhat|` scores, the finite-sample-corrected
  empirical quantile per step, a symmetric band around the point forecast clipped at 0 and the rated
  power;
* **CQR** for the kept trials with a quantile head: the score `max(q_lo - y, y - q_hi)` of the nominal
  outer quantiles and the band `[q_lo - Q, q_hi + Q]`, reported as level `trial_cqr` next to the split
  band of the same trial, so the value of the learned quantiles shows at equal coverage.

A coverage level is fitted only where there are enough calibration points for it: the corrected
quantile is the k-th smallest score with `k = ceil((n + 1) * coverage)`, which exists only for
`n >= coverage / (1 - coverage)` — 4 points for 80 %, 19 for 95 %, 99 for 99 % — and below
`EFORECAST_MIN_CALIBRATION_ROWS` (20) a quantile of a handful of residuals is not worth trusting.
A level that cannot be supported is left unfitted and said so in the log, where it used to be
returned as the largest residual seen, which is not a bound.

Outputs in `Results/`: `intervals.csv` (coverage achieved, mean width and interval (Winkler) score in
percent of rated, per predictor, level, slice and step), `intervals_summary.csv` (holdout, all steps,
sorted by interval score; shown on the Studies page), `interval_coverage.csv` (measured coverage on
the holdout per predictor of the `models`, `cluster_average` and `global` levels, broken down by
horizon step, by month and by regime — terciles of the observed target), `calibration.json` (the
corrections per predictor and coverage, in power units).

One band is a single correction for every hour of every season, so it over-covers the quiet periods
and under-covers the volatile ones; split conformal also assumes exchangeability, which a seasonal
series does not have. The breakdown does not repair either — nothing in a global correction can —
but it says where the band is not keeping its promise instead of averaging it away.  Online, `run_model_script.py --intervals 0.8 0.9` attaches
`lower_<c>_Step_k` / `upper_<c>_Step_k` columns from that file to the returned predictor
(`predict_online(..., intervals=[...])`).  Implementation: `combine_predictions/conformal.py`.
