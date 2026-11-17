# The pipeline
---

## 1. Preprocessing (`src/preprocessing.py`)

**Layer 1.** Before comparing apples and oranges you put them on the same scale. A
byte count of 1,000,000,000 and an error rate of 0.3 cannot sit in the same equation
untouched.

**Layer 2.** `build_preprocessor()` returns a scikit-learn `ColumnTransformer` with
three branches:

| Branch | Columns | Transform |
|---|---|---|
| `log` | 16 heavy-tailed counters (`LOG_FEATURES`) | `log1p(x)` then `MinMaxScaler` |
| `num` | the other 22 numeric columns | `MinMaxScaler` |
| `cat` | `protocol_type`, `service`, `flag` | `OneHotEncoder(handle_unknown="ignore")` |

Output: a float32 matrix of shape (n, 122). `log1p` is `log(1 + x)`, which maps 0 to 0
and compresses 1.4 billion to about 21. Min-max scaling: `(x - min) / (max - min)` with
min and max taken from the **training split only** (`preprocess()` calls
`fit_transform` on train and `transform` on test).

`handle_unknown="ignore"`: a service that appears only in the test set becomes an
all-zero indicator vector instead of crashing.

**Layer 3.** Why fit on train only? Fitting the scaler on test data leaks the test
distribution into the model (the max of a test column would be known). Why min-max and
not standardisation? The autoencoders and the logistic meta-learner want bounded
inputs; the tree models do not care. Why log1p? Without it the scaled `src_bytes` of
99.9% of rows would be below 0.001 and the autoencoder would learn nothing about them.
What breaks if you drop it: autoencoder reconstruction becomes dominated by the few
huge rows, and the anomaly score loses sensitivity.

---

## 2. HRA-FS: hybrid rank-aggregated feature selection (`src/feature_selection.py`)

**Layer 1.** Three judges rank the contestants. Instead of trusting one judge, you add
up the placings: a contestant who is 3rd, 5th and 4th beats one who is 1st, 80th and
90th. That sum of placings is a Borda count.

**Layer 2.** `hybrid_rank_selection(X, y, feature_names, k=40)` computes three
relevance scores per column using the 5-class training labels:

| View | Function | Captures | Known bias |
|---|---|---|---|
| Mutual information | `mutual_info_classif` (k-NN estimator, k=5) | any dependence, linear or not | noisy on sparse one-hot columns |
| ANOVA F-statistic | `f_classif` | separation of class means relative to within-class spread | linear only |
| Extra-Trees impurity importance | `ExtraTreesClassifier(200, balanced).feature_importances_` | interactions | favours high-cardinality continuous columns |

Each score vector is turned into a rank (1 = best) by `_rank`. Borda score:

```
B_j = (d - r_MI_j) + (d - r_F_j) + (d - r_ET_j),   d = 122
```

The 40 columns with the largest B_j are kept (`mask`). The whole table is saved to
`results/tables/feature_ranking.csv`. Top of the list: `dst_bytes`, `src_bytes`,
`dst_host_srv_count`, `dst_host_serror_rate`, `flag_S0`, `flag_SF`, `same_srv_rate`,
`count`, `logged_in`.

**Layer 3.** Why fuse? Each criterion fails differently; a feature that scores well
under only one view may be exploiting that view's bias. `same_srv_rate` is ranked 1st by
ANOVA-F but 24th by Extra-Trees; `dst_host_srv_count` is the reverse. Both survive
because they are in the upper part of every list. Why k = 40? The Borda curve flattens
there and it is close to the original 41 attributes. Effect (ablation A vs B): +1 point of
accuracy with two thirds of the columns removed, and 82 near-constant indicator columns
gone. If removed: nothing catastrophic, slightly slower and noisier models. Trade-off:
the ranking uses labels, so it selects for the *known* classes; that is exactly why the
autoencoder does **not** use the selected columns (next section).

---

## 3. Autoencoders as anomaly detectors (`src/autoencoder.py`)

**Layer 1.** Teach someone to draw only cats, from memory, by showing them thousands of
cats. Then hand them a photo of a dog and ask them to redraw it. The redrawing will be
bad. The badness of the redrawing tells you "this is not a cat", without ever having
shown them a dog.

**Layer 2.** `NormalProfileAutoencoder` wraps `MLPRegressor(hidden_layer_sizes=(32, 8, 32))`
trained to reproduce its input: topology 122 -> 32 -> 8 -> 32 -> 122, ReLU, Adam,
learning rate 1e-3, batch 256, up to 60 epochs with early stopping. Loss:

```
L = (1/|N|) * sum over benign x of || x - g(f(x)) ||^2
```

It is fitted **only on benign training rows** (67,343). The 8-unit bottleneck can only
store what benign traffic has in common. The anomaly score of any record is the
per-record mean squared reconstruction error:

```python
def score(self, X):
    recon = self.model.predict(X)
    return np.mean((X - recon) ** 2, axis=1)
```

In `train.py` the score is divided by its 99.9th percentile on the training split and
clipped to [0, 1], then appended as a 41st column. As a stand-alone detector
(`predict`), the threshold is the 95th percentile of benign training scores (0.00676).

Results of the stand-alone AE on KDDTest+: 86.72% accuracy, 83.74% detection, 9.33%
false alarms. Better than every supervised baseline.

**Layer 3.** Why benign-only? If you train on attacks too, the bottleneck learns to
reconstruct attacks and the error stops discriminating. Why the full 122 columns and
not the 40 selected ones? HRA-FS keeps what separates DoS from Probe; the autoencoder
needs what describes *normal*, a different question. On a 40k subsample: 84.8% (122
cols) vs 83.0% (40 cols). Why is a single-number score not enough? It cannot say *which
part* of the record is abnormal, which matters for distinguishing families (section 6).
What breaks without it: unseen-attack detection falls from 64.9% to about 49%
(ablation C vs F has no AE and loses 5 points of accuracy). Trade-off: it raises the
false-alarm rate from ~3% (trees) to ~9%, because benign-but-unusual traffic also
reconstructs badly.

---

## 4. Ensembles and stacked generalisation (`src/models.py`)

**Layer 1.** Four doctors look at the same patient. Majority vote treats them as
equals. Stacking hires a fifth doctor whose only job is to learn, from past cases,
which of the four to trust for which kind of patient.

**Layer 2.** Level-0 learners (`base_learners`):

| Learner | Strategy | Settings |
|---|---|---|
| Random Forest | bagging, greedy splits | 300 trees, `balanced_subsample` |
| Extra Trees | bagging, random split thresholds | 300 trees, balanced |
| XGBoost | level-wise gradient boosting | 400 rounds, depth 8, lr 0.08 |
| LightGBM | leaf-wise gradient boosting | 400 rounds, 63 leaves, lr 0.05, balanced |

Diversity is the point: bagging reduces variance, boosting reduces bias, random splits
decorrelate the trees further.

Stacking (Wolpert 1992): split the training set into K = 5 folds. For each fold, fit
every learner on the other four and predict class probabilities for the held-out fold.
Concatenating gives the **out-of-fold (OOF)** meta-feature vector per training row:

```
u_i = [ P_RF(c|x_i) for all c, P_ET(...), P_XGB(...), P_LGBM(...), anomaly features ]
```

A logistic regression (balanced, C = 1) is trained on (u_i, y_i). Then every learner is
refitted on the full training set for inference. Dimensions: binary, 4 learners x 2
classes + 31 = 39 meta-features; 5-class, 4 x 5 + 1 = 21.

OOF is essential: if the meta-learner were trained on in-sample predictions, it would
see learners that are nearly perfect on their own training data and learn nothing
useful.

**Layer 3.** Why logistic regression as the meta-learner? Its coefficients are
readable (`results/tables/meta_learner_weights_binary.csv`): RF +5.4, ET +8.0, XGB -0.6,
LGBM +0.5, and named subspace errors like S2 +5.8. An interviewer can ask "what did the
stack learn" and you can answer with numbers. Why not vote? Voting gives every learner
equal influence on every record; boosting and bagging make different mistakes. Why not
a GBM meta-learner? More capacity, no interpretability, and with 21-39 inputs a linear
model is enough. What breaks without stacking: ablation B vs C, 77.9 -> 79.2.

---

## 5. Novelty-aware (dual-regime) stacking: the core idea

**Layer 1.** You are training the fifth doctor (the meta-learner) with case files. If
every case file shows the four doctors being right, the fifth doctor learns "always
agree with them" and never learns when to use the lab test (the anomaly score). So you
also show case files where the four doctors were confidently wrong because the disease
was new to them, and in those files the lab test was the only clue. Now the fifth
doctor learns: when they are unsure, trust the lab.

**Layer 2.** First run of the project (standard stacking): the base learners were
99.8% correct on OOF rows, the meta-learner gave the anomaly score a coefficient of
0.98, and the stack scored 80.1% on KDDTest+, *below* XGBoost alone (81.0%) and far below
the AE alone (85.8%).

Fix, in `NoveltyAwareStacking.fit`: build the meta-training set in two passes.

1. **Familiar regime**: ordinary `StratifiedKFold(5)`.
2. **Novel regime** (`_fold_ids` with `novelty_aware=True`): every *attack type* is
   assigned to exactly one fold by a greedy balanced partition (largest type first, to
   the fold with the least load); benign rows are spread uniformly. When fold k is held
   out, the learners have never seen the attack types it contains, so their OOF
   probabilities for those rows are what they would output for a zero-day.

The two meta-sets are concatenated (2n rows) and the logistic regression is fitted on
both. Fold assignment is saved to `results/tables/novelty_fold_assignment.csv`
(e.g. neptune -> fold 0, satan -> fold 1, ipsweep -> 2, portsweep -> 3, smurf -> 4).

Result: meta-weight on the anomaly score 0.98 -> 5.29; binary accuracy 81.25 -> 84.11
(ablation E vs F); unseen-attack detection 49 -> 65%.

A subtlety in `_oof`: a training fold in the novel regime may lack a whole class (all
U2R types in one fold), so labels are re-encoded per fold and XGBoost's contiguous-label
requirement is handled by `searchsorted` on `classes_`.

**Layer 3.** Why both regimes and not only the novel one? With only grouped folds the
meta-learner sees learners that are often wrong and over-corrects; at test time the
refitted learners are confident on the familiar 83% of traffic. The familiar regime
calibrates that case. Why is this novel? Group-wise CV for *evaluation* is common;
using attack-type-grouped folds to *build a stacking meta-set* so that a combiner
learns deferral behaviour is not, to our knowledge, in the NIDS literature. What
breaks without it: the anomaly feature is ignored (ablation E). Trade-off: training
cost doubles (each learner is fitted 11 times), and the meta-learner's probabilities
become less calibrated (notes 03, E7).

---

## 6. Subspace clustering and the KitNET-style profile (`src/subspace_autoencoder.py`)

**Layer 1.** One doctor looking at a whole patient says "something is off". Ten
specialists each looking at one organ say "heart fine, lungs fine, liver abnormal".
The second report is more useful for deciding what is wrong.

**Layer 2.** Adapted from Mirsky et al.'s Kitsune (NDSS 2018). Steps in `KitNETLite.fit`:

1. `correlation_groups(X_benign, m=10)`: compute the 122 x 122 Pearson correlation
   matrix on benign rows, convert to distance `1 - r`, run average-linkage agglomerative
   clustering, and cut the dendrogram at the smallest number of clusters for which no
   cluster has more than m = 10 columns. Constant columns go into one extra group.
   Result: **31 subspaces** (`experiments/kitnet_groups_m10.json`). They are
   semantically coherent without supervision: {serror_rate, srv_serror_rate, flag_S0,
   S1, S2} is the SYN-flood subspace; {num_compromised, num_root, root_shell,
   su_attempted, ..., service_telnet} is the shell/root subspace; 47 services that never
   occur in benign traffic form one group.
2. One tiny autoencoder per subspace: hidden size `ceil(0.75 * dim)`, trained on
   benign rows.
3. `group_scores(X)`: per subspace, the RMSE of reconstruction, min-max scaled by its
   benign training range and clipped to [0, 1]. Output shape (n, 31). This is the
   **anomaly profile**.
4. An output autoencoder over the 31 normalised RMSEs gives KitNET's own single score
   (`score`), used only as a reference baseline.

In v2 the 31-column profile replaces the single AE score for the **binary** task:
input = 40 HRA-FS columns + 31 subspace errors = 71 columns, and all 31 are passed
through to the meta-learner (`n_passthrough=31`).

**Layer 3.** Why does the profile beat the single score? The meta-learner can now learn
rules like "SYN subspace abnormal -> attack even if the trees say normal" and "only the
service subspace is abnormal -> probably odd but benign traffic" (negative weights on
S12, S18, S23 do exactly this). Numbers: 84.11 -> 86.52 accuracy, 64.9 -> 79.6%
unseen-attack detection, same false-alarm rate. Why is KitNET *alone* weak here
(66.5%)? Its output layer is built to learn temporal co-variation of subspace errors in
packet streams; i.i.d. connection records give it nothing to learn. Why m = 10? m = 5
makes 50 subspaces, half of them single columns whose one-neuron autoencoder learns
nothing (83.7%); m = 20 is the same as m = 10 within noise. Why not for 5-class? Thirty-one
extra meta-features for a 5-class logistic regression with 995 R2L and 52 U2R rows
shifts rare-class decisions without new label information: 78.5 -> 76.8%. Trade-off:
the meta-learner's probabilities saturate (notes 03, E7).

---

## 7. Class imbalance handling

**Layer 1.** If one answer is right 99% of the time, a lazy student always gives it.
You stop that by making the rare answers worth more marks.

**Layer 2.** `class_weight="balanced"` sets each class's weight to
`n / (n_classes * n_c)`, so U2R rows (52) weigh about 1,300x a Normal row in the loss.
Random Forest uses `balanced_subsample` (recomputed per bootstrap). The meta-learner is
balanced too. No oversampling (SMOTE) is used.

**Layer 3.** Effect: U2R recall 77.6% in the 5-class model, where Shone et al. report
2.7%. Cost: U2R precision is 2.4%, because the meta-learner routes 43% of novel R2L rows
into U2R. From an analyst's view those are still alerts of a rare, severe family; from
the benchmark's view they are R2L errors. Why not SMOTE? Synthetic rows interpolated
between 52 U2R samples are not credible connections, and it adds a dependency.

---

## 8. Evaluation protocol (`src/evaluate.py`, `src/train.py`)

- Train on all of KDDTrain+, evaluate **once** on all of KDDTest+. No test data is used
  for scaling, ranking, clustering, thresholds or model selection.
- Hyper-parameters were fixed a priori or chosen on a 40,000-row subsample of the
  training split.
- Reported: binary and 5-class metrics for 10 baselines and the proposed model;
  per-family detection; unseen-attack detection; KDDTest-21; ablation A-G; 5-fold CV;
  meta-learner weights; detection at FAR budgets; timing (14.6 microseconds per
  record at inference, about 68,000 records/s on one core).
- Every baseline receives the *same input matrix* as the proposed model for that task.

Interview line: "the test split was touched exactly once, by the final evaluation".

---