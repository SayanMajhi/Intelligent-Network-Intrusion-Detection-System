# Codebase
```
.
├── src/                    the pipeline (library modules + four entry points)
├── experiments/            paper-inspired experiments and their logged results
├── results/tables/         every number in the report, as CSV/JSON
├── docs/report/            REPORT.md, NIDS_Report.docx, figures/, build_docx.py
├── docs/notes/             research_notes.md, the lab notebook
├── docs/papers/            the three PDFs
├── data/                   KDDTrain+.txt, KDDTest+.txt
├── notes/                  these study guides
├── .github/workflows/ci.yml  smoke check on every push
├── requirements.txt, LICENSE, .gitignore, README.md
```

Running conventions: every entry point is run from the repository root as
`python src/<file>.py`; Python puts `src/` on `sys.path`, so modules import each other
by bare name (`from config import ...`). `experiments/run_experiments.py` inserts
`src/` into `sys.path` itself.

---

## src/config.py: the single source of constants

| Name | Value | Used by |
|---|---|---|
| `ROOT`, `DATA_RAW`, `RESULTS`, `FIGURES`, `TABLES`, `MODELS` | paths; `FIGURES = docs/report/figures` | everything |
| `TRAIN_FILE`, `TEST_FILE` | `data/KDDTrain+.txt`, `data/KDDTest+.txt` | `data_loader` |
| `RANDOM_STATE` | 42 | every stochastic component |
| `COLUMNS` | the 41 feature names + `label` + `difficulty` | `data_loader`, `preprocessing` |
| `CATEGORICAL` | `protocol_type`, `service`, `flag` | one-hot branch |
| `LOG_FEATURES` | 16 heavy-tailed counters | log1p branch |
| `ATTACK_CATEGORY` | 39 attack names -> {DoS, Probe, R2L, U2R}, normal -> Normal | `data_loader` |
| `CLASS_ORDER` | Normal, DoS, Probe, R2L, U2R | plots, tables |
| `N_SELECTED_FEATURES` | 40 | HRA-FS k |
| `AE_HIDDEN`, `AE_EPOCHS` | (32, 8, 32), 60 | single autoencoder |
| `STACK_CV_FOLDS` | 5 | stacking folds per regime |
| `SUBSPACE_MAX_SIZE` | 10 | Kitsune's m |

The module also creates the output directories on import. To run on another dataset
you change `COLUMNS`, `CATEGORICAL`, `LOG_FEATURES`, `ATTACK_CATEGORY` and the file
paths; nothing else refers to NSL-KDD by name.

---

## src/data_loader.py

- `load_split(path)`: `pd.read_csv(header=None, names=COLUMNS)`, then adds `category`
  (via `ATTACK_CATEGORY`, unknown names default to R2L) and `is_attack` (label !=
  "normal"). Returns a DataFrame of shape (n, 45).
- `load_nsl_kdd()`: returns `(train, test)`.
- `__main__`: prints shapes, class counts and the 17 attack types present only in test.

---

## src/preprocessing.py

- `FEATURE_COLUMNS`: the 41 features. `NUMERIC_LOG` (16) and `NUMERIC_PLAIN` (22)
  partition the numeric ones.
- `build_preprocessor()`: `ColumnTransformer` with branches `log` (FunctionTransformer
  log1p + MinMaxScaler), `num` (MinMaxScaler), `cat` (OneHotEncoder, unknown ignored).
- `get_feature_names(pre)`: the 122 output names in order (`log` cols, `num` cols,
  then one-hot names like `service_http`, `flag_S0`).
- `preprocess(train_df, test_df)`: fit on train, transform both, return
  `(X_train, X_test, names, preprocessor)` as float32.

---

## src/feature_selection.py

- `_rank(scores)`: stable argsort -> rank 1 = highest score.
- `hybrid_rank_selection(X, y, feature_names, k=40)`: mutual information (k-NN, k=5),
  ANOVA F (NaN/inf -> 0), Extra-Trees importance (200 trees, balanced); three ranks;
  Borda score `3d - sum of ranks`; returns a boolean `mask` over the 122 columns and the
  full ranking table (saved as `results/tables/feature_ranking.csv`).

---

## src/autoencoder.py

`NormalProfileAutoencoder`:
- `__init__`: `MLPRegressor(hidden_layer_sizes=(32, 8, 32), relu, adam, lr 1e-3, batch 256,
  max_iter 60, early_stopping, n_iter_no_change 8)`.
- `fit(X_normal, quantile=0.95)`: trains X -> X on benign rows; sets `threshold_` to the
  95th percentile of training errors.
- `score(X)`: per-row mean squared reconstruction error, shape (n,).
- `predict(X)`: 1 if score > threshold (stand-alone detector).
- `loss_curve_`: training loss per epoch (plotted in `fig_autoencoder.png`).

---

## src/subspace_autoencoder.py

- `correlation_groups(X, m=10)`: drops constant columns into a separate group, computes
  `1 - corrcoef`, average-linkage `linkage`, loops `fcluster(maxclust=n)` upward until
  every cluster has at most m columns. Returns a list of index arrays.
- `KitNETLite(m=10, beta=0.75, epochs=40)`:
  - `_make_ae(dim)`: `MLPRegressor` with one hidden layer of `ceil(beta * dim)` units.
  - `fit(X_normal)`: clusters, trains one AE per group, records per-group RMSE min/max
    on benign rows, trains the output AE over the normalised RMSE vector, sets a 95th
    percentile threshold on the output score.
  - `_group_rmse(X)`: (n, k) raw RMSE per subspace. Note the `reshape(len(X), -1)`:
    single-column groups make `predict` return 1-D, which would otherwise broadcast
    into an n x n matrix (a 17 GB crash we hit once).
  - `group_scores(X)`: the (n, k) normalised profile used by the v2 binary model.
  - `score(X)`: KitNET's own output-layer RMSE, (n,).
  - `predict(X)`: threshold on `score`, stand-alone baseline only.

---

## src/models.py

- `base_learners(n_classes)`: the four level-0 estimators with their settings.
- `NoveltyAwareStacking(BaseEstimator, ClassifierMixin)`:
  - Parameters: `n_classes`, `n_folds=5`, `novelty_aware=True`, `ae_passthrough=True`,
    `n_passthrough=1` (how many trailing columns of X go straight to the meta-learner),
    `random_state`.
  - `_stratified_folds(y)`: fold id per row from `StratifiedKFold`.
  - `_fold_ids(y, groups)`: with `novelty_aware`, greedy balanced partition of attack
    types into folds (largest type first); benign rows round-robin. Stores
    `fold_assignment_`.
  - `_oof(X, y, fold, C)`: for each fold, re-encode labels present in the training part
    (XGBoost needs contiguous 0..C-1), fit a clone of each learner, write its
    `predict_proba` into the right columns of the (n, 4C) OOF matrix via `searchsorted`.
  - `fit(X, y, groups=None)`: OOF in the familiar regime; if novelty-aware and groups
    given, OOF again in the novel regime and stack the two meta-sets (2n rows); fit the
    balanced `LogisticRegression(C=1, max_iter=3000)`; refit all four learners on all
    data into `estimators_`.
  - `_meta_features(oof, X)`: appends the last `n_passthrough` columns of X.
  - `_level0(X)`: concatenated `predict_proba` of the refitted learners.
  - `predict_proba`, `predict`: meta-learner on `[level0, passthrough]`.
  - `meta_weights()`: names and coefficients for the weights tables.
- `build_aese(n_classes, novelty_aware, ae_passthrough, n_passthrough)`: constructor
  helper used by `train.py`, the ablation and the experiments.
- `baseline_models(n_classes)`: the ten baselines (LR, NB, DT, KNN, LinearSVC, MLP, and
  the four tree learners by pretty name).

---

## src/evaluate.py

- `compute_metrics(y_true, y_pred, average)`: accuracy, precision, recall, F1, MCC; for
  binary also `detection_rate` and `false_alarm_rate` from the confusion matrix.
- `plot_confusion`, `plot_roc`, `plot_bar`: the figure helpers; `PALETTE` is shared.

---

## src/train.py: the experiment, top to bottom

1. Load data; class-distribution table and figure; list the unseen attack types.
2. Preprocess -> 122 columns; label-encode the five classes (alphabetical:
   DoS=0, Normal=1, Probe=2, R2L=3, U2R=4, which once caused a mislabelled confusion
   matrix, now fixed by using `le.classes_`).
3. HRA-FS -> `mask`, `X_train_sel` (n, 40); feature-ranking figure.
4. Single AE on benign rows of the full 122 columns; scaled score appended ->
   `X_train` (n, 41) for the 5-class task; AE figure; stand-alone AE metrics.
   4b. `KitNETLite(m=10)` on benign rows; `g_train` (n, 31); `X_train_b` (n, 71) for the
   binary task; subspace groups JSON; subspace-profile heatmap; stand-alone KitNET metrics.
5. `run_models(...)`: fits the ten baselines and the proposed model (with
   `groups=groups_train` so the novelty-aware folds know each row's attack type);
   binary on `X_train_b` with `n_pass=31`, 5-class on `X_train` with `n_pass=1`. Writes
   `results_binary.csv`, `results_multiclass.csv`, per-class report, unseen-attack table,
   KDDTest-21 table, operating points.
6. Ablation A-G; 5-fold CV of the binary model; confusion matrices, ROC, bar charts,
   ablation and CV figures; meta-learner weight tables; fold assignment; `joblib` dump of
   the whole pipeline to `results/models/aese_nids.joblib`; `summary.json`.

Runtime on a laptop CPU: about 45 minutes (binary stack 263 s, 5-class 394 s, CV 16 min).

---

## src/compare.py

`LITERATURE`: published KDDTest+ accuracies with their reference numbers. `main()`
reads our proposed rows from the result CSVs, writes `literature_comparison.csv` and
`fig_literature_comparison.png`.

## src/predict.py

Loads the joblib artefact, rebuilds both inputs (`X_mc` = selected + single AE score;
`X_bin` = selected + subspace profile), writes a CSV with attack probability, binary
decision, predicted family and the raw anomaly score. Usage:
`python src/predict.py data/KDDTest+.txt --out predictions.csv`.

## src/system_diagram.py

Draws `fig_system_diagram.png` with matplotlib patches. No logic.

---

## experiments/run_experiments.py

- Setup: load, preprocess, HRA-FS, single AE (+ bottleneck `encode()` for E2),
  `test21` mask, `unseen` mask.
- `log(...)`: appends one row to `experiments/results.csv`.
- `evaluate(...)`: metrics on KDDTest+ and on KDDTest-21 (`E*/E1` rows).
- `run(...)`: fits the novelty-aware stack for the requested tasks with a given
  `n_pass`.
- Blocks E0, E2, E3/E4/E5 (shared `KitNETLite` fit; writes `kitnet_groups_m10.json`),
  E6 (m in {5, 20}), E7 (operating points for E0 vs E3 -> `operating_points.csv`).
- Command-line arguments select experiments: `python experiments/run_experiments.py E3 E7`.

## experiments/results.csv

Columns: `exp, timestamp, paper, config, task, test_set, accuracy, precision, recall,
f1, mcc, far, unseen_dr, train_s, notes`. 26 rows.

---

## results/tables (what each file answers)

| File | Question |
|---|---|
| `results_binary.csv`, `results_multiclass.csv` | every model's metrics and timings |
| `per_class_report_aese.csv` | precision/recall/F1 per family, 5-class |
| `unseen_attack_detection.csv` | detection on the 17 unseen types |
| `results_test21.csv` | every model on the hard subset |
| `operating_points.csv` | detection at 1/2/5/10% FAR |
| `ablation.csv` | configurations A-G |
| `cv_binary.csv` | the 5 CV folds |
| `feature_ranking.csv` | all 122 features with the three scores, ranks, Borda |
| `subspace_groups.json` | the 31 subspaces |
| `meta_learner_weights_binary.csv`, `_multiclass.csv` | what the stack trusts |
| `novelty_fold_assignment.csv` | which attack type was held out in which fold |
| `class_distribution.csv`, `literature_comparison.csv`, `summary.json` | tables 1 and 10, headline numbers and timings |

---

## docs/

- `docs/report/REPORT.md`: the full report; `NIDS_Report.docx` is built from it by
  `build_docx.py` (a small Markdown-to-Word converter: headings, tables, images, code).
- `docs/notes/research_notes.md`: the lab notebook written during the project; the
  narrative source for Acts 2-4.
- `docs/papers/`: the three PDFs.

## .github/workflows/ci.yml

On every push: install requirements, byte-compile, import every module, load the
dataset, fit the whole pipeline on a 5,000-row subsample (3 epochs, k=20) and assert
accuracy above 0.5 on 500 test rows. About 2 minutes.

---

## Where is X? Quick index

| I want to change... | Go to |
|---|---|
| number of selected features, fold count, seed | `src/config.py` |
| the fold scheme | `NoveltyAwareStacking._fold_ids` in `src/models.py` |
| which learners are stacked | `base_learners` in `src/models.py` |
| the autoencoder shape | `AE_HIDDEN` in `src/config.py` |
| subspace size | `SUBSPACE_MAX_SIZE` in `src/config.py` |
| a metric | `compute_metrics` in `src/evaluate.py` |
| the attack-to-family mapping | `ATTACK_CATEGORY` in `src/config.py` |
| a literature number | `LITERATURE` in `src/compare.py` |
