# Intelligent Network Intrusion Detection System Using Machine Learning

An end-to-end NIDS built on the NSL-KDD benchmark. The proposed model, **AESE-NIDS**
(Autoencoder-augmented Stacked Ensemble), combines:

1. **HRA-FS** – Hybrid Rank-Aggregated Feature Selection (mutual information + ANOVA-F +
   Extra-Trees importance fused with a Borda count, top-40 features kept);
2. a **normal-profile autoencoder** trained on benign traffic only, whose reconstruction
   error is appended as an anomaly-score feature so the classifier can flag attacks it
   never saw during training;
3. a **stacked heterogeneous ensemble** (Random Forest, Extra Trees, XGBoost, LightGBM)
   whose out-of-fold probabilities are combined by a logistic-regression meta-learner.

Both binary (Normal vs Attack) and 5-class (Normal, DoS, Probe, R2L, U2R) detection are
evaluated on the official KDDTest+ split and compared with ten classical baselines,
an ablation study, 10-fold cross-validation, and results reported in the literature.

## Project structure

```
NIDS-ML/
├── data/
│   ├── raw/            KDDTrain+.txt, KDDTest+.txt (NSL-KDD)
│   └── processed/      (generated)
├── src/
│   ├── config.py           paths, column names, attack→category map, hyper-parameters
│   ├── data_loader.py      loads NSL-KDD and attaches binary / 5-class targets
│   ├── preprocessing.py    log1p + min-max + one-hot (fit on train only)
│   ├── feature_selection.py HRA-FS hybrid rank aggregation
│   ├── autoencoder.py      normal-profile autoencoder anomaly scorer
│   ├── models.py           baselines + proposed stacked ensemble
│   ├── evaluate.py         metrics, confusion matrix, ROC, bar charts
│   ├── train.py            full experiment runner (writes all tables/figures)
│   ├── compare.py          comparison with published results
│   ├── system_diagram.py   draws the block diagram
│   └── predict.py          scores new records with the saved model
├── results/
│   ├── tables/         CSV tables (metrics, ablation, CV, feature ranking, ...)
│   ├── figures/        PNG figures
│   ├── models/         aese_nids.joblib (trained pipeline)
│   └── train_log.txt   console log of the last run
├── report/
│   ├── REPORT.md       the full report (markdown source)
│   ├── NIDS_Report.docx the full report (Word, 10 pt single spaced)
│   ├── build_docx.py   markdown → docx converter
│   └── figures/        figures embedded in the report
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Dataset (NSL-KDD) is already in `data/raw/`. To re-download:

```bash
curl -L -o data/raw/KDDTrain+.txt https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
curl -L -o data/raw/KDDTest+.txt  https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt
```

## Run

```bash
python src/train.py            # ~10-15 min on a laptop CPU; writes results/ and report/figures/
python src/compare.py          # literature comparison table + chart
python src/system_diagram.py   # block diagram
python report/build_docx.py    # rebuild the Word report from REPORT.md
python src/predict.py data/raw/KDDTest+.txt --out predictions.csv
```

## Key results (KDDTest+)

See `results/tables/results_binary.csv`, `results_multiclass.csv`, `ablation.csv`,
`literature_comparison.csv` and the report for the full numbers.
