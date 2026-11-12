# Network Intrusion Detection with an Autoencoder-Augmented Stacked Ensemble

A machine-learning intrusion detector for connection records, evaluated on the
official NSL-KDD train/test split. Benign-only autoencoders produce anomaly features
(a single reconstruction error, or a 31-dimensional per-subspace error profile), and a
stacked ensemble of four tree learners combines them through a logistic meta-learner
whose training set is built by holding out whole attack types. On KDDTest+ the binary
detector reaches 86.5% accuracy and flags 79.6% of records from the 17 attack types
absent from training; the best single tree baseline on the same input reaches 79.8% and
41.2%.

## Pipeline

```
KDDTrain+ / KDDTest+ (41 features, 125,973 / 22,544 records)
        |
        v
preprocessing.py        log1p on 16 counters, min-max scaling, one-hot protocol/service/flag -> 122 columns
        |
        +------------------------------------------------------------+
        v                                                            v
feature_selection.py                                     autoencoder.py / subspace_autoencoder.py
HRA-FS: mutual information, ANOVA-F and                  trained on benign records only
Extra-Trees ranks fused by Borda count -> 40 columns     single AE 122-32-8-32-122 -> 1 error score (5-class task)
        |                                                 31 subspace AEs (correlation clustering, m=10) -> 31 errors (binary task)
        +------------------------------------------------------------+
        v
models.py  NoveltyAwareStacking
   level 0: RandomForest, ExtraTrees, XGBoost, LightGBM
   out-of-fold probabilities from two fold schemes: stratified, and one fold per attack type
   level 1: logistic regression on [probabilities, anomaly features]
        |
        v
Normal / Attack           and           Normal / DoS / Probe / R2L / U2R
```

The attack-type-grouped folds are the point of the design. With ordinary stratified
folds the base learners are almost always right on the meta-training rows, so the
meta-learner ignores the anomaly features. Holding out whole attack types gives it rows
where the trees are confidently wrong and the anomaly features are high, and it learns
to defer to them. Details, equations and algorithms are in Section 3 of the report.

## Repository layout

```
.
├── src/
│   ├── config.py                 paths, column names, attack-family map, hyper-parameters
│   ├── data_loader.py            reads NSL-KDD, adds binary and 5-class targets
│   ├── preprocessing.py          ColumnTransformer fitted on the training split only
│   ├── feature_selection.py      HRA-FS rank aggregation
│   ├── autoencoder.py            single benign-profile autoencoder
│   ├── subspace_autoencoder.py   KitNET-style subspace autoencoder ensemble
│   ├── models.py                 baselines and NoveltyAwareStacking
│   ├── evaluate.py               metrics and plots
│   ├── train.py                  full experiment: tables, figures, saved model
│   ├── compare.py                comparison with published NSL-KDD results
│   ├── predict.py                scores a CSV with the saved model
│   └── system_diagram.py         draws the block diagram
├── experiments/
│   ├── run_experiments.py        paper-inspired experiments E0-E7
│   ├── results.csv               one row per experiment run
│   ├── operating_points.csv      detection rate at fixed false-alarm budgets
│   └── kitnet_groups_m10.json    the 31 feature subspaces
├── results/tables/               metrics, ablation, per-class, KDDTest-21, meta weights
├── docs/
│   ├── report/                   REPORT.md, NIDS_Report.docx, figures/, build_docx.py
│   ├── notes/research_notes.md   design history: first version, papers, experiments, v2
│   └── papers/                   the three papers the experiments are based on
├── data/                         KDDTrain+.txt, KDDTest+.txt (NSL-KDD, 22 MB)
├── requirements.txt
└── README.md
```

`results/models/` (the trained pipeline) is created by `train.py` and is not tracked.

## Setup and execution

Python 3.10 or newer, CPU only.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Reproduce every table and figure (about 45 minutes on a laptop CPU):

```bash
python src/train.py
python src/compare.py
```

Re-run the paper-inspired experiments (about one hour; appends to `experiments/results.csv`):

```bash
python experiments/run_experiments.py
```

Score new records with the saved model (NSL-KDD column format, label columns optional):

```bash
python src/predict.py data/KDDTest+.txt --out predictions.csv
```

Rebuild the Word report from `REPORT.md`:

```bash
python docs/report/build_docx.py
```

Hyper-parameters (number of selected features, autoencoder shape, subspace size, fold
count, seed) are in `src/config.py`. The random seed is fixed; a run reproduces the
numbers below.

## Results on KDDTest+

| Model | Binary accuracy | Detection rate | False-alarm rate | Unseen-attack detection |
|---|---|---|---|---|
| Random Forest | 77.2 | 62.1 | 2.7 | 26.4 |
| XGBoost | 79.8 | 66.7 | 2.8 | 41.2 |
| MLP | 81.2 | 72.8 | 7.8 | - |
| Single autoencoder, unsupervised | 86.7 | 83.7 | 9.3 | - |
| Proposed, single AE score (v1) | 84.1 | 78.5 | 8.5 | 64.9 |
| Proposed, subspace profile (v2) | 86.5 | 83.0 | 8.8 | 79.6 |

Five-class: 78.5% accuracy, macro-F1 60.8 (highest of the compared models). The
five-class model keeps the single AE score; the subspace profile lowers its accuracy
to 76.8% because the rare families have too few training rows for 31 extra
meta-features.

Two limitations are measured rather than hidden. On the hard KDDTest-21 subset the
false-alarm rate is 38.7%. The v2 meta-learner's probabilities are saturated, so at a
false-alarm budget of 1-2% the boosting baselines rank attacks better
(`experiments/operating_points.csv`, Section 4.10 of the report).

## Documentation

- Report (abstract, literature review, method, results, comparison, references, code listing):
  [docs/report/REPORT.md](docs/report/REPORT.md), [docs/report/NIDS_Report.docx](docs/report/NIDS_Report.docx)
- Design history and experiment log, written as working notes:
  [docs/notes/research_notes.md](docs/notes/research_notes.md)
- Experiment rows: [experiments/results.csv](experiments/results.csv)
- Papers: [Tavallaee et al. 2009](docs/papers/Tavallaee2009_NSLKDD_CISDA.pdf) (NSL-KDD),
  [Shone et al. 2018](docs/papers/Shone2018_NDAE_TETCI.pdf) (stacked non-symmetric autoencoders),
  [Mirsky et al. 2018](docs/papers/Mirsky2018_Kitsune_NDSS.pdf) (Kitsune)

## Dataset

NSL-KDD (Tavallaee et al., 2009): 41 connection features, 39 attack labels grouped into
DoS, Probe, R2L and U2R. KDDTest+ contains 17 attack types that do not occur in
KDDTrain+ (3,750 records), and R2L is three times more frequent in test than in train.
The `difficulty` column is used to form KDDTest-21. Source:
`https://github.com/defcom17/NSL_KDD`.
