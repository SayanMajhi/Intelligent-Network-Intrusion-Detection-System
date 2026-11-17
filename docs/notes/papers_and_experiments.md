# Papers,experiments and the v1 to v2 evolution

Study guide for Acts 3 and 4. The project went through one design, a reading of three
papers, seven logged experiments, and one change. This file explains each paper, what
mechanism was taken from it and why that mechanism and not the whole method, and then
every experiment with its numbers. All numbers are in `experiments/results.csv` and
`results/tables/`.

---

## Part A. The three papers

### A.1 Tavallaee, Bagheri, Lu, Ghorbani (2009): "A detailed analysis of the KDD CUP 99 data set"

**What it is.** The paper that created NSL-KDD. PDF: `docs/papers/Tavallaee2009_NSLKDD_CISDA.pdf`.

**Layer 1.** They audited the old exam, found it full of repeated questions, and wrote
a fairer one. They also wrote down, for each question, how many of 21 students got it
right.

**Layer 2.** Findings: 78% of KDD'99 training rows and 75% of test rows were
duplicates; learners memorised frequent records. They removed duplicates, ran 21
learners (7 algorithms x 3 seeds), recorded `#successfulPrediction` per record (our
`difficulty` column), sampled the new test set inversely to it, and defined
KDDTest-21 as the records not all 21 got right. Their baselines on KDDTest+: NB 76.6,
SVM 69.5, MLP 77.4, J48 81.1, RF 80.7, NB-Tree 82.0; on KDDTest-21 everything fell to
the 50s-60s.

**Layer 3, what we took and why.** The *evaluation discipline*: report KDDTest-21 too,
and treat accuracy on KDDTest+ as a generalisation measure rather than a modelling one.
We did not take any model from it (theirs are 2009 baselines). Experiment E1 is the
direct consequence.

### A.2 Shone, Ngoc, Phai, Shi (2018): "A deep learning approach to network intrusion detection"

**What it is.** Stacked non-symmetric deep autoencoders (NDAE) for feature learning,
Random Forest on top. IEEE TETCI. PDF: `docs/papers/Shone2018_NDAE_TETCI.pdf`.

**Layer 1.** Instead of feeding raw features to the classifier, first let a network
compress them into a shorter summary, then classify the summary.

**Layer 2.** NDAE = encoder only (no decoder), sigmoid, three hidden layers each as
wide as the input; two stacked; trained unsupervised; the encoded representation goes
to a Random Forest because, in their words, softmax on top of stacked autoencoders is
weak compared with RF/KNN/SVM. GPU TensorFlow. They drop attack types with fewer than 20
records (so their test set is 18,794 rows, not 22,544). NSL-KDD 5-class: 85.42% overall,
DoS 94.6, Normal 97.7, Probe 94.7, **R2L 3.8, U2R 2.7**. Their 10-fold CV was 99.6%:
the same train/test gap we see.

**Layer 3, what we took and why.** The question "does the *code* (compressed
representation) carry information the *error* throws away?" became experiment E2. We
did not take their architecture because (a) their autoencoders are trained on all
traffic including attacks, which is a representation-learning use, while ours are
benign-only anomaly scorers, and (b) their R2L/U2R collapse (3.8% / 2.7%) shows that
representation learning alone does not solve the rare families; class balancing and an
anomaly signal do more (our R2L 39%, U2R 78% recall). Also note for comparison tables:
their 85.4% is on a reduced test set.

### A.3 Mirsky, Doitshman, Elovici, Shabtai (2018): "Kitsune: an ensemble of autoencoders for online network intrusion detection"

**What it is.** Unsupervised, online NIDS that runs on a Raspberry Pi. NDSS. PDF:
`docs/papers/Mirsky2018_Kitsune_NDSS.pdf`.

**Layer 1.** Instead of one big detector looking at everything, use many tiny
detectors, each watching a small group of related measurements, plus one more detector
watching how the tiny detectors' alarms normally move together.

**Layer 2.** KitNET:
- *Feature Mapper*: cluster the n features by correlation distance (1 - Pearson r)
  with agglomerative clustering; break the dendrogram until no cluster exceeds m
  features (m <= 10). Each cluster is a subspace.
- *Ensemble layer*: one three-layer autoencoder per subspace, hidden size
  ceil(beta * dim) with beta = 3/4, trained on benign traffic only, each emitting its
  RMSE.
- *Output layer*: an autoencoder over the 0-1 normalised vector of subspace RMSEs; its
  RMSE is the anomaly score. It learns how subspace errors co-vary, so "subspace 3 off
  while 7 is fine" is a stronger anomaly than the sum of both.
- Threshold phi: max training score, or a fitted log-normal. They report at a fixed
  false-positive rate (0.001) rather than at accuracy.
- Complexity: one big autoencoder is O(n^2), KitNET O(k^2) with k subspaces.

**Layer 3, what we took and why.** The *subspace decomposition* and the *per-subspace
RMSE vector*, and the habit of reporting detection at a false-alarm budget. We did not
take the output layer as our detector: on i.i.d. connection records it has no temporal
co-variation to learn and scores only 66.5% alone (E4u). What we did instead, feeding
the 31 subspace errors to a supervised meta-learner that already knows when to trust an
anomaly signal, is the combination that is not in the paper and is the project's second
contribution.

### A.4 Why "take a mechanism, not a method"

None of the three methods worked as a drop-in: Shone's code helped binary and hurt
5-class; Kitsune alone was weak; Tavallaee's hard subset mainly showed our false alarms.
The transferable units were smaller than the papers: an evaluation subset, a question
about representations, a decomposition of the feature space. The interview version:
"I read for mechanisms I could test, then let the logged experiment decide."

---

## Part B. The experiments (`experiments/run_experiments.py`)

Common setup: full KDDTrain+/KDDTest+, seed 42, same 40 HRA-FS columns, same
novelty-aware stack; only the anomaly features change. "Unseen DR" = detection rate on
the 3,750 rows of the 17 attack types absent from training. Each run appends a row to
`experiments/results.csv`; `E*/E1` rows are the same model scored on KDDTest-21.

### E0: baseline (the v1 pipeline)

| Task | Test set | Acc | DR | FAR | Unseen DR |
|---|---|---|---|---|---|
| binary | KDDTest+ | 84.11 | 78.54 | 8.53 | 64.85 |
| binary | KDDTest-21 | 70.01 | 71.60 | 37.17 | 64.76 |
| 5-class | KDDTest+ | 78.52 | macro-F1 60.79 | | |
| 5-class | KDDTest-21 | 59.68 | macro-F1 52.02 | | |

### E1: KDDTest-21 (from Tavallaee)

Free experiment: `difficulty < 21`, 11,850 rows. The lesson is the 37% false-alarm rate
on the hard subset: benign rows that 2009 learners also got wrong are, almost by
definition, unusual-looking benign traffic, and an anomaly-driven detector flags them.
Tavallaee's RF/J48 were in the 60s on Test-21; 70% is fine, but we now report both sets.

### E2: bottleneck code as features (from Shone)

The 8 ReLU activations of the single autoencoder's bottleneck (122 -> 32 -> 8), scaled by
their training maximum, appended next to the error score.

| | Binary acc | Unseen DR | 5-class acc | 5-class macro-F1 |
|---|---|---|---|---|
| E0 | 84.11 | 64.85 | 78.52 | 60.79 |
| E2 | 84.94 | 70.37 | 76.02 | 56.34 |

Binary up slightly, unseen detection +5.5 points: the code carries something the scalar
error loses. 5-class down 2.5 points: the code is a *benign* representation; it says how
a record differs from normal, not which family it is, so for 5-class it is 8 columns of
noise. Not adopted.

### E3: subspace anomaly vector (from Kitsune) - the one that worked

31 subspace RMSEs (m = 10) passed through to the meta-learner, replacing the single
score.

| | Binary acc | DR | FAR | Unseen DR | 5-class acc |
|---|---|---|---|---|---|
| E0 single AE error | 84.11 | 78.54 | 8.53 | 64.85 | 78.52 |
| E4u KitNET-lite alone (unsupervised, 95% threshold) | 66.51 | 43.65 | 3.27 | 41.65 | - |
| E4 KitNET output score as the single feature | 85.61 | 81.38 | 8.81 | 73.31 | 77.78 |
| **E3 31 subspace RMSEs as meta-features** | **86.52** | **83.00** | 8.84 | **79.63** | 76.84 |
| E5 E3 + the old single error | 85.34 | 80.85 | 8.72 | 72.16 | 77.33 |

Three findings. (1) KitNET alone is weak here but its subspace errors as features are
the best thing tried: +2.4 accuracy, +4.5 detection, +15 on unseen attacks, same FAR.
(2) Adding the old single score on top (E5) makes it worse: the vector already contains
that information and the extra column is one more thing for the logistic regression to
fit. (3) 5-class drops 1.7 points for the same reason as E2.

### E6: subspace size m

| m | subspaces | Binary acc | Unseen DR |
|---|---|---|---|
| 5 | 50 | 83.71 | 63.89 |
| 10 | 31 | 86.52 | 79.63 |
| 20 | 20 | 86.57 | 79.57 |

m = 5 is worse than no vector at all (half the subspaces are single columns; a
one-neuron autoencoder of one feature learns nothing). m = 10 and 20 are equal within
noise; m = 10 kept because the groups are more readable and it is Kitsune's own bound.

### E7: detection at a false-alarm budget (Kitsune's way of reporting)

Threshold set so at most x% of benign test rows are flagged:

| FAR budget | Random Forest | XGBoost | LightGBM | E0 single score | E3 / v2 |
|---|---|---|---|---|---|
| 1% | 45.4 | 45.3 | 44.3 | 41.1 | 0.0 |
| 2% | 50.8 | 50.6 | 48.4 | 52.5 | 0.0 |
| 5% | 78.5 | 84.3 | 86.8 | 67.1 | 67.4 |
| 10% | 87.0 | 86.0 | 93.1 | 93.9 | 86.3 |

The uncomfortable result, kept on purpose. The v2 meta-learner's probabilities are
**saturated**: with 31 passthrough columns and class-balanced logistic regression, a
subset of benign rows gets probability 1.0, so no threshold reaches a 1-2% FAR, and at
10% the boosting models rank better. v2 wins at its default operating point and on
novel attacks but is a worse *ranker*. First item of future work: calibrate the
meta-learner (Platt scaling on the novel regime, or a stronger L2 penalty).

Interview framing: "accuracy at threshold 0.5 and ranking quality are different
properties; our model is better at the first and worse at the second, and the table
says so."

---

## Part C. Ablation (from `src/train.py`, `results/tables/ablation.csv`)

| Config | Binary acc | Binary F1 | 5-class acc | 5-class macro-F1 |
|---|---|---|---|---|
| A: all 122 features, Random Forest | 76.89 | 75.17 | 73.66 | 47.17 |
| B: HRA-FS (40), Random Forest | 77.95 | 76.58 | 73.82 | 48.85 |
| C: HRA-FS, standard stacking, no anomaly feature | 79.20 | 78.21 | 79.09 | 56.54 |
| D: HRA-FS + AE score, Random Forest | 78.65 | 77.46 | 73.82 | 48.88 |
| E: HRA-FS + AE score, standard stacking | 81.25 | 80.76 | 80.11 | 58.59 |
| F: HRA-FS + AE score, novelty-aware stacking (v1; final 5-class) | 84.11 | 84.91 | 78.52 | 60.79 |
| G: HRA-FS + subspace profile, novelty-aware stacking (v2; final binary) | 86.52 | 87.51 | 76.84 | 56.95 |

Reading the binary column: feature selection +1.1, stacking +1.3, anomaly score +2.0,
novelty-aware folds +2.9 (largest single step), subspace profile +2.4. Reading 5-class:
the pattern holds through F, then G costs 1.7 accuracy and 3.8 macro-F1, so the 5-class
model stops at F.

---

## Part D. Changelog

| Version | Change | Binary acc | Unseen DR |
|---|---|---|---|
| v1.0 | HRA-FS + single AE error + standard stacking | 80.1 | ~49 |
| v1.1 | novelty-aware (dual-regime) stacking | 84.1 | 64.9 |
| v1.2 | AE on all 122 columns instead of the 40 selected | (+1.8 on subsample) | |
| v2.0 | subspace anomaly profile (m = 10) for the binary task; 5-class unchanged | 86.5 | 79.6 |

Version history is also in `docs/notes/research_notes.md`, written at the time.

---

## Part E. Comparison with published work on KDDTest+

| Reference | Method | Binary | 5-class |
|---|---|---|---|
| Tavallaee 2009 | NB Tree (best classical) | 82.02 | - |
| Javaid 2016 | self-taught learning (sparse AE + softmax) | 88.39 | 79.10 |
| Yin 2017 | RNN-IDS | 83.28 | 81.29 |
| Shone 2018 | stacked NDAE + RF (reduced test set) | - | 85.42 |
| Su 2020 | BAT-MC (BLSTM + attention) | - | 84.25 |
| Ieracitano 2020 | statistical analysis + autoencoder | 87.00 | - |
| This work | AESE-NIDS v2 | 86.52 | 78.52 |

Honest positioning: above every classical learner and RNN-IDS on binary, within 0.5
to 1.9 points of the two autoencoder-based deep models; below the deep models on
5-class. Unlike all of them we report unseen-attack detection, KDDTest-21 and FAR
budgets, train in minutes on a CPU, and have a 39-coefficient readable meta-learner.
Also say: the published numbers are quoted from the papers and protocols differ
slightly.

---