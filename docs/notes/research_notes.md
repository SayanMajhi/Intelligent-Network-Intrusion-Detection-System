# Research notes – how this project actually went
---

## 1. Where I started (Week 1)

The first version of the pipeline (`src/train.py`) looked good on paper: one-hot, log, and min-max preprocessing, Borda-count feature selection (keeping the top 40 features via mutual info, ANOVA F, and Extra-Trees), a benign-only autoencoder (AE) providing reconstruction error as an extra feature, and a stacked ensemble (RF, ExtraTrees, XGBoost, LightGBM) with a logistic regression meta-learner.

The results on `KDDTest+` (binary) were surprising:

| Model | Accuracy | Detection Rate |
| --- | --- | --- |
| XGBoost alone | 81.0% | 68.7% |
| My stack + AE feature | 80.1% | 67.1% |
| Autoencoder alone (unsupervised) | 85.8% | 81.8% |

My supervised stack performed *worse* than XGBoost alone, while an unsupervised autoencoder beat everything. That turned out to be the most useful finding of the project. The real challenge on this dataset isn't separating known classes (5-fold CV on the training set hits 99.8%)—it's catching the 17 novel attack types (3,750 rows) that don't appear in the training set at all.

Inspecting the meta-learner showed it gave the AE score a weight of just 0.98, essentially ignoring it. With standard cross-validation, the base trees see every attack type and hit ~99.8% accuracy out-of-fold, so the meta-learner learns to just copy them. It never sees a case where the trees are confidently wrong, so it never learns when to trust the AE.

To fix this, I built the meta-training set twice: once with standard stratified folds, and once holding out *entire attack types* per fold (e.g., `neptune` in fold 0, `satan` in fold 1). In that second pass, the trees are blind to the held-out attacks, forcing the meta-learner to learn the pattern: *"trees say normal, AE says weird, label says attack."* This pushed the AE weight from 0.98 to 5.3, binary accuracy from 80.1% to 84.1%, and unseen-attack detection from 49% to 65%—what I called "novelty-aware stacking" in the report.

One smaller finding: the AE performs better on all 122 features than on the 40 selected ones (84.8% vs. 83.0% on a subsample). Feature selection keeps what separates *known* attacks, while the AE needs the full context of what defines *normal* traffic.

That's where things stood when I wrote the report—before I went back and started reading the literature properly.

---

## 2. Papers (week 2)

I picked three with different angles: the dataset paper, a supervised deep-learning
paper that uses an autoencoder for representation, and an unsupervised paper that uses
an *ensemble* of autoencoders. For each I wrote down what they did, what I believed,
and what I wanted to try because of it.

### 2.1 Tavallaee, Bagheri, Lu, Ghorbani – "A detailed analysis of the KDD CUP 99 data set" (CISDA 2009)

The NSL-KDD paper. Things I hadn't appreciated before reading it:

- 78% of the KDD'99 training set and 75% of the test set were duplicate rows. That's why
  old papers report 99% on KDD'99: the model memorises frequent records and gets asked
  about them again.
- They ran 21 learners (7 algorithms × 3 random seeds) and recorded for every record how
  many of the 21 got it right. That count is the `difficulty` column at the end of every
  NSL-KDD row (the 43rd field I had been throwing away).
- They built KDDTest+ by sampling inversely to that difficulty, so easy records are
  under-represented, and **KDDTest-21** = the test records that *not all* 21 learners got
  right (difficulty < 21). 11,850 rows. They say to report on both.
- Their baselines on KDDTest+: NB 76.6, SVM 69.5, MLP 77.4, J48 81.1, RF 80.7, NB-Tree
  82.0. On KDDTest-21 everything drops to the 50s-60s.

What I took from it: (a) I should report KDDTest-21 as well, it's free since I have the
difficulty column; (b) the 85% ceiling everyone hits is by construction, not bad
modelling; (c) accuracy on this split is mostly "how many of the deliberately hard rows
do you get", which is a generalisation question. That matched what I'd found.

-> Experiment E1: score every model on KDDTest-21 too.

### 2.2 Shone, Ngoc, Phai, Shi – "A deep learning approach to network intrusion detection" (IEEE TETCI 2018)

Non-symmetric deep autoencoder (NDAE): encoder only, no decoder, sigmoid, three hidden
layers each as wide as the input. Two of them stacked, trained unsupervised, and then the
encoded representation is classified by a Random Forest because, in their words, a
softmax on top of stacked autoencoders is weak compared with RF/KNN/SVM. TensorFlow on a
GTX 750. They drop attack types with < 20 records.

Their NSL-KDD 5-class numbers (their Table 4): overall 85.42% accuracy, but
DoS 94.6, Normal 97.7, Probe 94.7, **R2L 3.8, U2R 2.7**. They openly say R2L/U2R are
"anomalous" and blame the small training counts (995 and 52). Also notable: their 10-fold
CV on NSL-KDD was 99.6%, same train/test gap I saw.

What I took from it:
- The "autoencoder as a feature learner, then trees" pattern is exactly the half of my
  design I had, but I only use the *error*. They use the *code* (the compressed
  representation). Maybe the code carries information the error throws away.
- Their R2L/U2R collapse is worse than mine (mine: R2L 39%, U2R 78% recall, though with
  bad U2R precision). The class-balanced weights and the anomaly feature are doing
  something there. Worth keeping.
- They evaluate 5-class on all of KDDTest+ but their test counts (18,794) show they
  dropped rows, so their 85.4% isn't on quite the same set as my 78.5%. Noted for the
  comparison table.

-> Experiment E2: add the 8-dim bottleneck activations of my AE as classifier features,
alongside the error.

### 2.3 Mirsky, Doitshman, Elovici, Shabtai – "Kitsune: an ensemble of autoencoders for online network intrusion detection" (NDSS 2018)

Completely unsupervised, online, runs on a Raspberry Pi. The core (KitNET):

- Feature Mapper: cluster the n features by *correlation distance* (1 − Pearson r),
  agglomerative, cut the dendrogram until no cluster has more than m features (m ≤ 10 in
  their experiments). Each cluster is a "subspace".
- Ensemble layer: one tiny three-layer autoencoder per subspace, hidden size
  ceil(β·dim) with β = 3/4, trained only on benign traffic, each outputs its RMSE.
- Output layer: another small autoencoder over the 0-1 normalised vector of subspace
  RMSEs. Its RMSE is the final anomaly score. Point of the second layer: it learns how the
  subspace errors normally co-vary, so "subspace 3 is off *while* subspace 7 is fine" can
  be flagged even if each alone is within range.
- Threshold φ: max score seen in training, or fit a log-normal. They evaluate at a fixed
  FPR of 0.001 rather than at accuracy, which I think is the right way to talk about an
  IDS.
- Complexity argument: one big AE is O(n²), KitNET is O(k²) with k the number of
  subspaces.

What I took from it:
- My single 122→8→122 autoencoder gives the classifier one number: "how weird is this
  connection". KitNET gives a *vector*: how weird is the byte-count subspace, how weird is
  the SYN-error subspace, how weird is the login/content subspace. For 5-class that might
  matter a lot: R2L should look abnormal in content/login features and normal in traffic
  features; DoS the opposite. One number can't say which.
- Their threshold-at-fixed-FPR framing. My meta-learner outputs a probability, so I can
  report detection rate at a false-alarm budget (1%, 2%, 5%, 10%) instead of one accuracy.
- Their point that supervised NIDS need experts to label everything, which is why an
  unsupervised signal should be *inside* the model. That was my original motivation, so
  good to see it argued by people who shipped it.

-> Experiments E3/E4/E5/E6: re-implement KitNET offline (`src/subspace_autoencoder.py`),
feed the per-subspace RMSE vector to the meta-learner (E3), try the KitNET output score
as a drop-in for my single AE score (E4), both together (E5), sweep m (E6).
-> Experiment E7: detection rate at fixed false-alarm budgets.

### 2.4 One I couldn't get

I wanted Yin et al. 2017 (RNN-IDS, IEEE Access) as well since it's the standard deep
baseline (83.3 binary / 81.3 5-class on KDDTest+, 122 one-hot features, no feature
selection). IEEE Xplore blocks scripted downloads and I only had the numbers from
secondary sources, so I cite it in the comparison table but did not base an experiment
on it.

---

## 3. Experiments (week 2-3)

Everything below is in `experiments/results.csv` (one row per run, appended by
`experiments/run_experiments.py`). All on the full KDDTrain+ / KDDTest+ split, same
seed, same 40 HRA-FS features and the same novelty-aware stack; only the anomaly
features change. "unseen DR" = fraction of the 3,750 test rows from the 17 attack types
not in training that get flagged. Train time is for the whole stack on my laptop.

### 3.1 E1 – scoring on KDDTest-21 (Tavallaee)

Free experiment: KDDTest-21 is just KDDTest+ rows with `difficulty < 21` (11,850 rows,
52.6%). My baseline pipeline:

| | KDDTest+ | KDDTest-21 |
|---|---|---|
| binary accuracy | 84.11 | 70.01 |
| binary detection rate | 78.54 | 71.60 |
| false-alarm rate | 8.53 | 37.17 |
| 5-class accuracy | 78.52 | 59.68 |

The false-alarm rate on Test-21 is the eye-opener: 37% of the *normal* rows in the hard
subset get flagged. Those are benign connections that even in 2009 looked like attacks to
most learners, and my anomaly score is doing exactly what it's built to do on them. So on
the hard subset I'm trading false alarms for recall. Tavallaee's RF/J48 were in the 60s on
Test-21 so 70% is fine, but I'm now reporting both sets everywhere.

### 3.2 E2 – Shone-style bottleneck code as features

Took the 8 ReLU activations of my AE's bottleneck (122→32→8), scaled by their training
max, and appended them to the 40 features next to the error score.

| | binary acc | unseen DR | 5-class acc | 5-class macro-F1 |
|---|---|---|---|---|
| E0 baseline | 84.11 | 64.85 | 78.52 | 60.79 |
| E2 + code(8) | 84.94 | 70.37 | 76.02 | 56.34 |

Binary: small gain, and unseen-attack detection up 5.5 points, so the code does carry
something the scalar error doesn't. 5-class: 2.5 points worse. My reading: the code is a
*benign* representation (trained only on normal rows), so it describes how a record
differs from normal but not what family it is, and for the 5-class meta-learner it's 8
more columns of mostly-noise. Shone et al. train their NDAEs on everything (attacks
included) and classify the code directly, which is a different use. Kept as a result,
not adopted.

### 3.3 E3 – Kitsune-style subspace anomaly vector (the one that worked)

Re-implemented KitNET offline (`src/subspace_autoencoder.py`): correlation-distance clustering of the 122 columns on
benign training rows, cut until no group has more than m = 10 features → 31 subspaces;
one small AE per subspace (hidden = ceil(0.75·dim)); 0-1 normalised RMSE per subspace;
an output AE over the 31 RMSEs for the KitNET score. The groups it found make sense on
their own (`experiments/kitnet_groups_m10.json`):

- {serror_rate, srv_serror_rate, flag_S0, flag_S1, flag_S2} – the SYN-flood subspace
- {rerror_rate, srv_rerror_rate, dst_host_rerror_rate, ..., flag_REJ, flag_RSTO, flag_RSTR} – the port-scan subspace
- {num_compromised, num_root, num_file_creations, num_access_files, root_shell, su_attempted, service_telnet} – the "someone got a shell" subspace
- {urgent, num_failed_logins}, {hot, is_guest_login, service_ftp} – R2L-ish content
- {diff_srv_rate, protocol_type_icmp, service_eco_i, service_ecr_i, service_urp_i} – ICMP sweeps
- one 47-column group of services that never occur in benign training traffic

Then instead of one anomaly number, the meta-learner gets all 31 subspace RMSEs
(passed straight through, next to the base-learner probabilities).

| | binary acc | DR | FAR | unseen DR | 5-class acc |
|---|---|---|---|---|---|
| E0 single AE error | 84.11 | 78.54 | 8.53 | 64.85 | 78.52 |
| E4u KitNET-lite alone (unsupervised, 95% thr.) | 66.51 | 43.65 | 3.27 | 41.65 | – |
| E4 KitNET output score as the single feature | 85.61 | 81.38 | 8.81 | 73.31 | 77.78 |
| **E3 31 subspace RMSEs as meta-features** | **86.52** | **83.00** | 8.84 | **79.63** | 76.84 |
| E5 E3 + the old single AE error | 85.34 | 80.85 | 8.72 | 72.16 | 77.33 |

Three things I didn't expect:

1. KitNET-lite *alone* is a bad detector here (66.5%, worse than my single AE at 86.7%).
   Kitsune is built for packet streams where the output AE learns temporal co-variation of
   subspace errors; on i.i.d. connection records that second layer doesn't have much to
   learn. But the *subspace errors as features* are the best thing I tried: +2.4 accuracy,
   +4.5 detection rate, +15 points on unseen attacks, at the same false-alarm rate.
   The meta-learner can learn "SYN subspace off → attack even if the trees say normal"
   and "service subspace off but everything else fine → probably benign odd traffic",
   which a single global score can't express.
2. Adding my old single score on top (E5) makes it *worse* than E3. The vector already
   contains that information, and the extra column just gives the logistic regression
   one more thing to fit.
3. 5-class goes down (78.5 → 76.8). Same story as E2: 31 extra meta-columns for a
   5-class logistic regression with only ~1,000 R2L and 52 U2R training rows is too
   many; it shifts rare-class decisions around without new label information. The
   single-score version stays better for the family task.

### 3.4 E6 – how big should a subspace be?

| m | subspaces | binary acc | unseen DR |
|---|---|---|---|
| 5 | 50 | 83.71 | 63.89 |
| 10 | 31 | 86.52 | 79.63 |
| 20 | 20 | 86.57 | 79.57 |

m = 5 is worse than no vector at all: half the "subspaces" are single columns, and a
1-neuron autoencoder of one feature learns nothing. m = 10 and 20 are the same within
noise. I kept m = 10 (Kitsune's own upper bound) since the groups are more readable.

### 3.5 E7 – detection at a false-alarm budget (Kitsune's way of reporting)

Instead of accuracy at the default 0.5 threshold, detection rate when the threshold is
set so that at most x% of normal test rows are flagged:

| FAR budget | E0 single score | E3 subspace vector |
|---|---|---|
| 1 % | 41.1 % | 0.0 % |
| 2 % | 52.5 % | 0.0 % |
| 5 % | 67.1 % | 67.4 % |
| 10 % | 93.9 % | 86.3 % |

This one is uncomfortable and I'm keeping it in. The E3 meta-learner's probabilities are
saturated: with 31 passthrough columns and class-balanced logistic regression, a chunk
of *normal* test rows get probability 1.0, so there is no threshold that reaches a
1-2% false-alarm rate, and at 10% the single-score model actually detects more. So
E3 wins at the default operating point and on novel attacks, but it is a worse ranker.
If I had another week I'd calibrate the meta-learner (Platt scaling on the held-out
regime, or an L2 penalty sweep) before anything else.

### 3.6 What I'm keeping

- Binary detector: E3 (HRA-FS 40 + 31 subspace RMSEs + novelty-aware stack). Headline
  86.5% / 83.0% detection / 79.6% on unseen attacks.
- 5-class detector: the original single-score version (78.5%), because the vector hurts
  the rare families.
- Report both KDDTest+ and KDDTest-21, and the FAR-budget table, because after reading
  Tavallaee and Kitsune I don't think a single accuracy number says much on this data.

Honest overall take: none of the three paper ideas worked as a drop-in. Shone's code
helped binary and hurt 5-class; Kitsune's full model alone was weak here; Tavallaee's
hard subset mostly showed how many false alarms the anomaly view costs. What worked was
taking one *mechanism* (subspace-wise reconstruction errors) and plugging it into the
part of my pipeline that could use it (the meta-learner that already knew how to
weigh an anomaly signal thanks to the novelty-aware folds). I think that's the actual
contribution: not the subspace AE, not the stacking, but giving a learned combiner a
*structured* anomaly signal and training it in a way where it has a reason to use it.

---

## 4. Changelog of the main pipeline

- v1 (first run): HRA-FS + single AE error + standard stacking. 80.1% binary. Shelved.
- v1.1: novelty-aware (dual-regime) stacking. 84.1% binary, 64.9% unseen. In the report.
- v1.2: AE on all 122 columns instead of 40. (+1.8 on subsample.) In the report.
- v2 (after papers): subspace anomaly vector for the binary task, m = 10. 86.5% binary,
  79.6% unseen. 5-class unchanged at 78.5%. KDDTest-21 and FAR-budget tables added.
  `src/train.py` now runs v2; numbers in the report are regenerated from it.

