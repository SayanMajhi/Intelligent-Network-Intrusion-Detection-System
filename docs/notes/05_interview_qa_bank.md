# 05. Interview question bank

Twenty-five questions an examiner or interviewer could ask about this project, with
model answers. Each answer gives the number that backs it where one exists. Practise
saying them out loud in under a minute each.

---

### Conceptual

**1. What problem does this project solve, in one sentence?**
It detects malicious network connections from 41 summary features, with the specific
goal of still detecting attack types that were not in the training data, which is what
signature-based systems and plain supervised models fail at.

**2. Why NSL-KDD and not a modern dataset?**
It has an official train/test split with a deliberate distribution shift (17 attack
types only in test, R2L three times more frequent in test), which makes it the hardest
widely used test of generalisation; it is small enough for a full ablation on a CPU;
and it has the most published numbers to compare against. Its weakness is age: 1998
traffic, no modern attacks. The code is dataset-agnostic apart from `config.py`.

**3. Why is cross-validation accuracy on KDDTrain+ (99.7%) not the headline?**
Within KDDTrain+ every fold contains the same attack types, so CV measures separation of
known classes. KDDTest+ measures generalisation to unseen ones. The 13-point gap *is*
the result; reporting only CV would hide it.

**4. What is the difference between anomaly detection and misuse detection, and which is this?**
Misuse detection learns what attacks look like (supervised); anomaly detection learns
what normal looks like and flags deviations (unsupervised). This project is a hybrid
where the anomaly signal is a *feature inside* a supervised model, and the combiner
learns from data how much to trust it, instead of two separate systems merged by a rule.

**5. Explain the Borda count and why three ranking criteria instead of one.**
Each criterion ranks the 122 columns; a column's Borda score is the sum of (122 - rank)
over the three criteria; keep the top 40. Mutual information is non-linear but noisy on
sparse indicators, ANOVA-F is linear but stable, tree importance captures interactions
but favours high-cardinality columns. A column must do well under several views to
survive, which filters out each view's bias. `same_srv_rate` is 1st under ANOVA-F and
24th under Extra-Trees; it survives because it is high in all three.

**6. Why is the autoencoder trained only on benign traffic?**
Its bottleneck can only store what the training data has in common. Trained on benign
rows, it reconstructs benign traffic well and everything else badly, so reconstruction
error is "distance from normal" and needs no attack examples. Trained on attacks too,
it would reconstruct attacks well and the error would stop discriminating.

**7. Why does the autoencoder use all 122 columns when the classifier uses 40?**
Feature selection keeps columns that separate the *known* classes. The autoencoder's job
is to describe *normal*; a column that is useless for DoS-vs-Probe can still be
informative about normality. Measured on a 40k subsample: 84.8% with 122 columns vs
83.0% with 40.

**8. What is stacking and why use it instead of voting?**
Stacking trains a meta-learner on the out-of-fold predictions of the base learners.
Voting gives every learner equal weight on every record; the meta-learner learns which
learner to trust and, here, how much to trust the anomaly features. Our logistic
meta-learner's coefficients are readable: Extra Trees +8.0, Random Forest +5.4, the
ftp/bytes subspace error +5.8, XGBoost -0.6.

**9. What is an out-of-fold prediction and why does stacking need it?**
A prediction for a training row made by a model that did not train on that row. If the
meta-learner saw in-sample predictions, the base learners would look nearly perfect and
the meta-learner would learn to copy them, which is overfitting at the second level.

**10. What is novelty-aware stacking? This is the key question.**
Standard OOF folds are stratified, so the base learners have seen every attack type and
are 99.8% right on the meta-training rows; the meta-learner learns "copy the trees" and
gave the anomaly score a weight of 0.98. We build the meta-training set twice: once with
stratified folds, once with folds where every attack type is held out exactly once.
In the second pass the trees are confidently wrong on the held-out types while the
anomaly features are high, so the meta-learner learns to defer to them. Weight on the
anomaly score went to 5.3, binary accuracy from 81.3 to 84.1, unseen-attack detection
from 49 to 65%.

**11. Why both regimes and not only the attack-held-out one?**
At test time the refitted base learners are confident and correct on the 83% of
traffic that is familiar. A meta-learner trained only on the novel regime has seen
learners that are often wrong and would over-correct. The familiar regime teaches it
the normal case; the novel regime teaches it the exception.

**12. What is the subspace anomaly profile?**
The 122 columns are clustered by correlation into 31 groups of at most 10 (SYN-error
rates with S0/S1/S2 flags, REJ/RST flags with rejection rates, root/shell counters with
telnet, and so on). One small autoencoder per group, trained on benign rows, gives a
per-group reconstruction error. The classifier receives all 31, so it can see *which
part* of a connection is abnormal instead of one global number. Adapted from Kitsune.

**13. Why does the profile help binary but hurt 5-class?**
Binary: 84.1 -> 86.5 accuracy, 64.9 -> 79.6% unseen detection, because the meta-learner
can learn subspace-specific rules. 5-class: 78.5 -> 76.8, because 31 extra meta-features
for a five-class logistic regression trained on 995 R2L and 52 U2R rows shift the
rare-class decisions without adding label information. So each task uses the
representation that works for it, and the ablation table shows both.

---

### Mathematical

**14. Write the autoencoder loss and the anomaly score.**
Loss over benign rows: L = (1/|N|) sum ||x - g(f(x))||^2 with encoder f, decoder g,
topology 122-32-8-32-122. Score a(x) = (1/122) ||x - g(f(x))||^2, scaled by the 99.9th
percentile over training and clipped to [0, 1]. Subspace version: r_j(x) =
sqrt(mean over i in G_j of (x_i - x_hat_i)^2), min-max scaled per subspace.

**15. Write the logistic meta-learner and say what its inputs are.**
P(y = c | u) = exp(w_c . u + b_c) / sum_c' exp(w_c' . u + b_c'), with u = [P_RF(.|x),
P_ET(.|x), P_XGB(.|x), P_LGBM(.|x), anomaly features]. Binary: 8 + 31 = 39 inputs.
5-class: 20 + 1 = 21. Balanced class weights, L2 with C = 1.

**16. Define detection rate, false-alarm rate and MCC, and say which matters most to whom.**
DR = TP/(TP+FN); FAR = FP/(FP+TN); MCC = (TP*TN - FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)).
Security teams care about DR (misses are breaches); operations care about FAR (too many
alarms and nobody reads them); MCC is the single number that is honest under imbalance.

**17. Why macro-averaged F1 for the 5-class task?**
Macro averages weight each class equally. With micro or weighted averaging, a model
that ignores R2L (2,885 rows) and U2R (67 rows) still looks fine. Macro-F1 falls from
60.8 to 57.0 when the subspace profile is added to the 5-class model; accuracy would
have hidden most of that.

**18. The v2 model detects 0% of attacks at a 1% false-alarm budget but 86.5% accuracy. How?**
Accuracy is measured at the default threshold 0.5; the FAR-budget table moves the
threshold. The v2 meta-learner's probabilities are saturated: with 31 passthrough
columns and balanced logistic regression a subset of benign rows gets probability
exactly 1.0, so no threshold can exclude them while keeping any attack. The model is a
good classifier at its operating point and a poor ranker. The fix is calibration
(Platt scaling on the novel regime or a stronger L2 penalty), listed as first future work.

---

### What-if and trade-off

**19. What happens if you remove the novelty-aware folds?**
Ablation E vs F: binary accuracy 81.25 -> 84.11 with them. Without them the anomaly
feature is effectively ignored and the stack behaves like its best member.

**20. What happens if you remove the autoencoder entirely?**
Ablation C (stacking, no anomaly feature): 79.2% binary. The unsupervised AE alone
scores 86.7%, which says the dominant error source on this benchmark is novelty, not
noise.

**21. The single autoencoder alone (86.7%) beats your full model (86.5%). Why keep the full model?**
Three reasons. The AE gives one binary verdict at a fixed quantile and cannot assign a
family; the full model does both tasks. The full model's decision is learned and
adjustable; the AE's threshold is a quantile. And at the same accuracy the full model
has a lower false-alarm rate (8.8 vs 9.3). The important point is that both sit far
above every purely supervised learner.

**22. Your false-alarm rate on KDDTest-21 is 38.7%. Defend that.**
KDDTest-21 is the subset that at least one of 21 reference learners got wrong; its
benign rows are benign connections that look unusual. An anomaly-driven detector flags
unusual things by design. On the same subset the tree baselines have 12% FAR but only
50-56% detection; we have 77.5%. It is a trade-off, it is measured, and it is in the
report rather than hidden. Operating at a chosen FAR needs the calibration work in Q18.

**23. How would you deploy this, and what breaks?**
A flow exporter (Zeek or CICFlowMeter) in front, producing connection records; the
joblib pipeline scores 68,000 records per second on one core. What breaks: the benign
profile drifts as the network changes, so the autoencoders need periodic refitting on
recent benign traffic; the attack-type map is NSL-KDD-specific; and the threshold must
be set to the analyst's false-alarm budget, which needs calibration first.

**24. Why XGBoost gets a negative meta-weight (-0.57) in the binary stack?**
The four learners' probabilities are highly correlated; the logistic regression
assigns the shared signal to Extra Trees and Random Forest (+8.0, +5.4) and uses XGBoost
and LightGBM as small corrections. A negative coefficient on one of several collinear
inputs is not "XGBoost is harmful"; it is the meta-learner saying the marginal
information in XGBoost, given the others, points slightly the other way.

**25. What would you do with another month?**
In order: calibrate the meta-learner so the model can be operated at a chosen FAR;
validate on CIC-IDS2017 and UNSW-NB15; add payload- or sequence-level features for
R2L, whose test rows look like benign SNMP/HTTP at the connection level; try on-line
updating of the benign profile; and replace the greedy fold partition with one that
also balances families, to see whether 5-class can use the subspace profile.

---

## Rapid-fire numbers to memorise

| Fact | Value |
|---|---|
| Train / test rows | 125,973 / 22,544 |
| Encoded columns / selected / subspaces | 122 / 40 / 31 |
| Unseen attack types / rows | 17 / 3,750 |
| KDDTest-21 rows | 11,850 |
| v2 binary accuracy / DR / FAR | 86.52 / 83.00 / 8.84 |
| v2 unseen-attack detection | 79.63% (RF 26.4, XGB 41.2, v1 64.9) |
| 5-class accuracy / macro-F1 | 78.52 / 60.79 |
| Meta-weight on anomaly score, standard vs novelty-aware | 0.98 vs 5.29 |
| 5-fold CV on KDDTrain+ | 99.68% |
| Inference speed | 14.6 us/record, about 68k records/s |
| Training time, binary / 5-class | 263 s / 394 s |
