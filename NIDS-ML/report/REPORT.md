# Intelligent Network Intrusion Detection System Using Machine Learning: An Autoencoder-Augmented Stacked Ensemble with Hybrid Rank-Aggregated Feature Selection

## Abstract

Network intrusion detection systems (NIDS) are a central layer of defence for modern enterprises, yet signature-based products are blind to novel attacks and many published machine-learning detectors degrade sharply when they meet traffic that differs from their training data. This work presents AESE-NIDS, an Autoencoder-augmented Stacked Ensemble for intelligent intrusion detection. The method has three parts. First, a Hybrid Rank-Aggregated Feature Selection (HRA-FS) stage ranks every encoded feature under three complementary relevance views (mutual information, ANOVA F-statistic and Extra-Trees impurity importance) and fuses the three rankings with a Borda count, keeping the forty features that are consistently useful. Second, a bottleneck autoencoder is trained only on benign traffic over the full encoded representation; its per-record reconstruction error is appended to the selected features as an explicit anomaly score, giving the supervised stage a signal that does not depend on having seen a given attack before. Third, four heterogeneous tree ensembles (Random Forest, Extra Trees, XGBoost and LightGBM) are stacked with a logistic-regression meta-learner that receives their out-of-fold class probabilities together with the anomaly score. The meta-training set is built in two regimes, ordinary stratified folds and folds in which entire attack types are held out, so the meta-learner learns how the base learners behave both on familiar traffic and on attacks they have never seen, and when to defer to the anomaly score. The system is evaluated on the NSL-KDD benchmark using the official KDDTrain+/KDDTest+ split, which contains seventeen attack types that never appear in training. Results are reported for binary (Normal vs Attack) and five-class (Normal, DoS, Probe, R2L, U2R) detection against ten classical baselines, an ablation study, ten-fold cross-validation and results published in prior work. The proposed model achieves {{BIN_ACC}}% binary accuracy and {{MC_ACC}}% five-class accuracy on KDDTest+, with a detection rate of {{BIN_DR}}% and a false-alarm rate of {{BIN_FAR}}%, and detects {{UNSEEN_DR}}% of records belonging to attack types absent from the training set. The complete implementation is provided in Python with scikit-learn, XGBoost and LightGBM.

**Keywords:** intrusion detection, NSL-KDD, feature selection, autoencoder, stacked ensemble, anomaly detection, machine learning.

---

## 1. Introduction

### 1.1 Background and problem statement

Every organisation that connects to the Internet exposes a surface that adversaries probe continuously. Firewalls and access-control lists decide which connections are allowed, but they cannot judge whether an allowed connection is being used for a legitimate purpose. A network intrusion detection system (NIDS) fills that gap: it observes traffic that has already passed the perimeter and raises an alarm when the traffic looks like an attack. The classical way to build a NIDS is signature matching. A human analyst studies a known exploit, writes a rule that captures its byte pattern or protocol behaviour, and the sensor compares every packet against the rule set. Signature engines are precise and explainable, but they share one structural weakness: they can only recognise what somebody has already described. A zero-day exploit, a slightly mutated malware sample or a slow, low-volume reconnaissance scan produces no match and passes silently. At the same time the volume, speed and heterogeneity of traffic keep growing, so that a rule set large enough to be useful becomes expensive to maintain and slow to evaluate.

Machine learning offers a different contract. Instead of enumerating attacks, a model is shown examples of benign and malicious connections and learns a decision boundary from measurable properties of each connection: its duration, the number of bytes exchanged in each direction, the protocol and service, error-flag ratios, how many recent connections targeted the same host, and so on. If the boundary captures what makes malicious traffic different rather than what makes a particular exploit different, the model should generalise to attacks it has not seen. This promise has driven a large body of research since the late 1990s, and the NSL-KDD benchmark [1], a cleaned revision of the KDD Cup 1999 data, remains the most widely used reference for comparing such detectors.

The problem addressed in this work is therefore the following: given labelled network connection records, build a classifier that (a) separates benign from malicious traffic with high detection rate and low false-alarm rate, (b) assigns each malicious record to one of the four standard attack families (Denial of Service, Probe, Remote-to-Local and User-to-Root), and (c) keeps a useful share of that accuracy when confronted with attack types that were not in its training data, as happens in the official KDDTest+ evaluation split where seventeen of the attack labels never occur in KDDTrain+.

### 1.2 Research gap

Reading the literature on NSL-KDD reveals a consistent pattern. Reported accuracies on random hold-out splits of KDDTrain+ or on cross-validation routinely exceed 99%, while the same models fall to between 75% and 85% on KDDTest+. Tavallaee et al. [1] already documented this in the paper that introduced the dataset: six standard learners trained on the full training set scored between 69.5% and 82.0% on the test set. The gap is not a tuning artefact; it comes from the distribution shift that the test set deliberately introduces. Three specific gaps motivate the present work.

First, most studies treat feature selection as a single-criterion filter (one information-theoretic measure, one wrapper or one embedded importance). Each criterion has a known failure mode: mutual information is estimated from discretised or neighbour-based densities and is noisy for sparse one-hot columns; the ANOVA F-statistic only sees linear mean differences; tree impurity importance is biased toward high-cardinality continuous attributes. A feature that scores well under only one view may simply be exploiting that view's bias.

Second, supervised detectors and unsupervised anomaly detectors are usually evaluated as competitors rather than combined. Supervised tree ensembles dominate on the attack families that are well represented in training (DoS and Probe), but they have no mechanism for saying "this record looks like nothing I was trained on". Autoencoders trained on benign traffic do have such a mechanism, but used alone they produce too many false alarms to be deployed [12], [19].

Third, ensembles in this area are usually homogeneous (bagging of one tree type) or combined by majority vote. Voting assigns every learner the same influence on every record, even though a gradient-boosted model and a randomised forest make different kinds of mistakes. Stacked generalisation [16], in which a meta-learner is trained on the out-of-fold predictions of the base learners, is a well-understood way to learn those trust weights, but it is still rarely used in NIDS work and almost never combined with an explicit anomaly signal.

### 1.3 Proposed solution

This report proposes AESE-NIDS, an Autoencoder-augmented Stacked Ensemble. The pipeline, shown in Figure 1 of Section 3, has three stages that correspond directly to the three gaps above. (1) HRA-FS, a Hybrid Rank-Aggregated Feature Selection stage, computes three independent rankings of the encoded features (mutual information, ANOVA F and Extra-Trees importance) and fuses them with a Borda count, so that a feature must be useful under several views to survive. (2) A normal-profile autoencoder with a 122-32-8-32-122 topology is trained only on benign records of the training split, over the complete encoded representation rather than the selected subset, because columns that carry little supervised signal can still describe what benign traffic looks like; its reconstruction error for each record is scaled and appended to the selected features as a 41st input. The supervised learners can then use "distance from the benign manifold" as a feature, which is informative for attacks whose supervised signature was never learned. (3) Four heterogeneous tree ensembles are trained as level-0 learners and a logistic-regression level-1 model is fitted to their out-of-fold class probabilities together with the anomaly score. The out-of-fold predictions are generated in two regimes: ordinary stratified five-fold splitting, which shows the meta-learner how the base learners behave on familiar traffic, and attack-type-grouped splitting in which every attack type is held out exactly once, which shows it how they behave on attacks they have never seen. This dual-regime, novelty-aware stacking is what allows the meta-learner to learn when to defer to the anomaly score. Its coefficients are inspectable and show which signal the system has learned to trust.

### 1.4 Motivation

The design is motivated by three practical considerations. A deployable NIDS must have a low false-alarm rate, because an analyst who receives hundreds of false alerts per day stops reading them; this argues for a strong supervised backbone rather than a pure anomaly detector. It must also degrade gracefully on novel traffic, which argues for an anomaly signal inside the model rather than beside it. Finally, it must be reproducible and cheap to retrain: every component used here is available in open-source libraries, trains in minutes on a laptop CPU, and the entire experiment is a single script.

### 1.5 Contributions

The contributions of this work are:

1. A hybrid, rank-aggregated feature selection procedure (HRA-FS) that fuses three complementary relevance criteria through a Borda count and reduces the 122-column encoded NSL-KDD representation to 40 features without loss of accuracy.
2. A fusion scheme in which the reconstruction error of a benign-only autoencoder is injected as an explicit feature into a supervised ensemble, so that anomaly information and discriminative information are combined inside one model rather than by a hand-tuned rule.
3. Novelty-aware (dual-regime) stacking: a stacked heterogeneous ensemble of Random Forest, Extra Trees, XGBoost and LightGBM whose logistic-regression meta-learner is trained on out-of-fold probabilities from both stratified folds and attack-type-grouped folds, so that it learns to combine the base learners with the anomaly score on unfamiliar traffic. To our knowledge, holding out whole attack types to build the stacking meta-set has not been reported for NIDS.
4. A thorough evaluation on the official NSL-KDD split: binary and five-class results, ten classical baselines, an ablation isolating each component, five-fold cross-validation, explicit measurement of detection on the seventeen unseen attack types, timing, and comparison with published results.
5. A complete, documented, open implementation (Section 7) that regenerates every table and figure in this report.

The remainder of the report is organised as follows. Section 2 reviews related work. Section 3 describes the dataset, the proposed method and its mathematical formulation. Section 4 reports the experimental setup and results. Section 5 compares the results with previous work. Section 6 concludes, and Section 7 lists the references and the source code.

---

## 2. Literature Review

### 2.1 Benchmarks for learning-based intrusion detection

The KDD Cup 1999 dataset, derived from the DARPA 1998 evaluation, was the first large labelled corpus for intrusion detection, but it contained an enormous number of duplicated records that biased learners toward frequent attacks and inflated reported accuracies. Tavallaee et al. [1] analysed these defects and released NSL-KDD, which removes duplicates, balances the difficulty of records and provides a fixed training/testing split (KDDTrain+ and KDDTest+) in which the test set contains attack types absent from training. Their baseline experiments with Naive Bayes, SVM, a multi-layer perceptron, J48, Random Forest and NB Tree produced test accuracies between 69.5% and 82.0%, numbers that remain a sobering reference point for any new method. Later corpora address the age of the underlying traffic: UNSW-NB15 [3] was generated with a commercial traffic generator and contains nine modern attack families, and CIC-IDS2017 [2] captured five days of realistic traffic with attacks such as brute force, Heartbleed, botnets, DoS and DDoS, infiltration and web attacks. Surveys by Khraisat et al. [8], Buczak and Guven [9], Ferrag et al. [10] and Ahmad et al. [24] catalogue these datasets and the methods applied to them. NSL-KDD is used in this work because it is small enough to allow a complete, reproducible comparison of many models and because its official split is the hardest widely used test of generalisation to unseen attacks.

### 2.2 Classical machine learning detectors

A long line of work applies standard classifiers to connection-level features. Dhanabal and Shantharajah [23] compared J48, SVM and Naive Bayes on NSL-KDD and observed the strong advantage of tree-based models, which handle the mixed categorical and numeric attributes without distributional assumptions. Random Forest [13] became the de-facto strong baseline because bagging over randomised trees reduces variance and gives usable feature importances; Extremely Randomised Trees [17] push the randomisation further by drawing split thresholds at random, which lowers variance again at a small bias cost. Gradient boosting, in the XGBoost [14] and LightGBM [15] implementations, fits trees sequentially to the residual of the current ensemble and often gives the best single-model accuracy on tabular data. Sommer and Paxson [20] gave an influential critique of this whole area, arguing that the closed-world assumption behind supervised learning, the high cost of false positives and the semantic gap between a flagged record and an actionable alert explain why academic detectors rarely reach production. Their critique is the reason the present work reports false-alarm rate and detection on unseen attacks rather than accuracy alone.

### 2.3 Feature selection for NIDS

The 41 NSL-KDD attributes expand to more than 120 columns after one-hot encoding of protocol, service and flag, and many columns are nearly constant. Ambusaidi et al. [18] proposed a flexible mutual-information-based filter that handles both linear and non-linear dependence and reported improved accuracy and lower cost for an LS-SVM detector on KDD Cup 99, NSL-KDD and Kyoto 2006+. Zhou et al. [22] combined correlation-based feature selection with a bat-inspired search (CFS-BA) and fed the reduced set to an ensemble of C4.5, Random Forest and Forest-by-Penalising-Attributes combined by average-of-probabilities, achieving competitive accuracy on NSL-KDD, AWID and CIC-IDS2017. These works show that feature reduction rarely hurts and often helps, but each relies on one selection criterion. Rank aggregation, where several criteria vote on the order of features, is common in bioinformatics but has been little explored for intrusion detection, and it is the strategy adopted here.

### 2.4 Deep learning detectors

Deep models have dominated NSL-KDD publications since 2016. Javaid et al. [4] used self-taught learning: a sparse autoencoder learned a representation of the data without labels and a softmax classifier was trained on top, giving 88.4% binary and 79.1% five-class accuracy on KDDTest+. Yin et al. [5] trained a recurrent neural network (RNN-IDS) on the one-hot-encoded features and reported 83.3% binary and 81.3% five-class accuracy. Shone et al. [6] introduced non-symmetric deep autoencoders stacked for unsupervised feature learning with a Random Forest classifier on top, reporting 85.4% five-class accuracy on NSL-KDD and large gains on KDD Cup 99. Vinayakumar et al. [7] performed a broad comparison of deep neural networks against classical learners across several datasets and found that DNNs were competitive but not uniformly superior. Su et al. [11] proposed BAT-MC, a bidirectional LSTM with an attention layer over packet-level features, reaching 84.3% five-class accuracy on KDDTest+. Ieracitano et al. [12] combined a statistical analysis of the features with an autoencoder-based classifier and reported about 87% binary accuracy. Mirsky et al. [19] took the autoencoder idea to online operation with Kitsune, an ensemble of small autoencoders whose reconstruction errors are aggregated by a second autoencoder, and showed it detecting attacks against IoT devices with no labelled data at all. Two observations follow from this group of papers. First, five-class accuracy on KDDTest+ plateaus in the low-to-mid eighties regardless of architecture, because the R2L and U2R families are both rare in training and dominated in testing by unseen variants. Second, the autoencoder consistently appears as the component that contributes robustness to novelty, whether used for representation learning [4], [6] or for anomaly scoring [12], [19].

### 2.5 Ensemble and hybrid approaches

Ensembles are attractive for NIDS because different learners fail on different records. Most reported ensembles, including that of Zhou et al. [22], combine members by voting or probability averaging. Stacked generalisation [16] instead trains a second-level model on the predictions of the first level and has been shown in other domains to outperform voting when the members are diverse. Hybrid designs that pair an anomaly detector with a misuse detector have also been described, typically as two separate systems whose alerts are merged by a rule. The approach taken in this report differs in that the anomaly score becomes an input feature of the supervised ensemble, so the combination weights are learned from data, and in that the ensemble is stacked rather than voted.

### 2.6 Summary of the gap

Across the reviewed work, the three ingredients used here (multi-criterion feature selection, a benign-profile autoencoder and stacked heterogeneous trees) each appear individually, but their combination in one trainable pipeline evaluated on the official NSL-KDD split has not been reported. The proposed AESE-NIDS fills that gap, and Section 4 quantifies the contribution of each ingredient through ablation.

---

## 3. Material and Methods

### 3.1 Dataset

NSL-KDD provides 125,973 training records (KDDTrain+) and 22,544 test records (KDDTest+). Each record describes one TCP/IP connection with 41 attributes in four groups: basic attributes of the connection (duration, protocol, service, flag, bytes in each direction), content attributes derived from the payload (failed logins, root shell, file creations), time-based traffic attributes computed over a two-second window (counts of connections to the same host or service and the corresponding error rates) and host-based traffic attributes computed over the last 100 connections. A label names the specific attack (or "normal") and a difficulty score records how many of 21 reference learners classified the record correctly. The 39 attack labels are grouped into four families following the original DARPA taxonomy: Denial of Service (DoS), Probe, Remote-to-Local (R2L) and User-to-Root (U2R). Table 1 gives the class distribution.

| Class | KDDTrain+ | KDDTest+ |
|---|---|---|
| Normal | 67,343 | 9,711 |
| DoS | 45,927 | 7,460 |
| Probe | 11,656 | 2,421 |
| R2L | 995 | 2,885 |
| U2R | 52 | 67 |
| Total | 125,973 | 22,544 |

Table 1. Class distribution of the NSL-KDD split used in this work.

Two properties of the split drive the difficulty of the task. The R2L family is nearly three times larger in testing than in training, and U2R has only 52 training examples. Moreover, seventeen attack labels (apache2, httptunnel, mailbomb, mscan, named, processtable, ps, saint, sendmail, snmpgetattack, snmpguess, sqlattack, udpstorm, worm, xlock, xsnoop, xterm) appear only in KDDTest+. Those {{N_UNSEEN}} records are a built-in test of generalisation to novel attacks and are reported separately in Section 4.

### 3.2 System overview

![Figure 1. Block diagram of the proposed AESE-NIDS.](figures/fig_system_diagram.png)

Figure 1 shows the pipeline. Raw records are pre-processed (Section 3.3), reduced by HRA-FS (Section 3.4), augmented with the autoencoder anomaly score (Section 3.5) and classified by the stacked ensemble (Section 3.6). All fitting (scalers, encoder, feature ranking, autoencoder, base learners and meta-learner) uses the training split only; the test split is touched once, for the final evaluation.

### 3.3 Pre-processing

Let a record be the vector of raw attributes $x = (x_1, \dots, x_{41})$. Three attributes (protocol_type, service, flag) are nominal and are one-hot encoded; categories that occur only in the test split are mapped to the all-zero vector. Sixteen counter-type attributes such as duration, src_bytes, dst_bytes and count are heavy-tailed (src_bytes spans nine orders of magnitude), so they are compressed with $x' = \log(1 + x)$ before scaling. Every numeric column is then min-max scaled to $[0, 1]$ using the minimum and maximum observed in the training split:

$$\tilde{x}_j = \frac{x'_j - \min_{\text{train}} x'_j}{\max_{\text{train}} x'_j - \min_{\text{train}} x'_j}.$$

The encoded representation has $d = 122$ columns. Scaling to a common range matters for the mutual-information estimator, the autoencoder and the linear meta-learner; tree learners are invariant to it.

### 3.4 Hybrid Rank-Aggregated Feature Selection (HRA-FS)

HRA-FS computes three relevance scores for every column $j$ using the five-class training labels $y$.

**Mutual information.** $I(X_j; Y) = \sum_{y} \int p(x_j, y) \log \frac{p(x_j, y)}{p(x_j)\,p(y)}\, dx_j$, estimated with the Kraskov nearest-neighbour estimator ($k = 5$). MI captures arbitrary non-linear dependence.

**ANOVA F-statistic.** With class means $\bar{x}_{j,c}$, overall mean $\bar{x}_j$, $n_c$ records in class $c$ and $C = 5$ classes,

$$F_j = \frac{\sum_c n_c (\bar{x}_{j,c} - \bar{x}_j)^2 / (C - 1)}{\sum_c \sum_{i \in c} (x_{ij} - \bar{x}_{j,c})^2 / (n - C)}.$$

F measures how far the class-conditional means are separated relative to within-class spread; it is linear but very stable.

**Extra-Trees importance.** An Extremely Randomised Trees model with 200 trees and balanced class weights is fitted, and the importance of column $j$ is the total impurity decrease attributed to splits on $j$, averaged over the trees. This captures interactions between features.

Each score vector is converted to a rank $r^{(v)}_j \in \{1, \dots, d\}$ (rank 1 = most relevant) and the three rankings are fused with a Borda count:

$$B_j = \sum_{v \in \{\text{MI}, F, \text{ET}\}} \left(d - r^{(v)}_j\right).$$

The $k = 40$ columns with the largest $B_j$ are retained. Borda fusion rewards consistency: a column ranked 5th, 7th and 6th outranks one ranked 1st, 80th and 90th. The value $k = 40$ was chosen because the Borda curve flattens there (Figure 3 in Section 4) and because it is close to the original 41-attribute dimensionality. Algorithm 1 summarises the procedure.

```
Algorithm 1: HRA-FS
Input : training matrix X (n x d), labels y, number of features k
Output: boolean mask M of length d
1  s_MI <- mutual_info(X, y)                 # k-NN estimator, k = 5
2  s_F  <- anova_f(X, y)                     # replace NaN/inf by 0
3  s_ET <- ExtraTrees(200 trees, balanced).fit(X, y).feature_importances_
4  for v in {MI, F, ET}:  r_v <- rank(s_v)   # 1 = best, stable ties
5  B <- (d - r_MI) + (d - r_F) + (d - r_ET)  # Borda count
6  M <- indices of the k largest B
7  return M
```

### 3.5 Normal-profile autoencoder

Let $\mathcal{N}$ be the set of benign training records in the full encoded space, $x \in [0,1]^{122}$. The autoencoder deliberately uses all 122 columns rather than the 40 selected ones: HRA-FS keeps the columns that discriminate the *known* classes, whereas the autoencoder has to model what benign traffic looks like, and a column that is useless for discriminating DoS from Probe may still be informative about normality (a preliminary experiment on a 40,000-record subsample gave 84.8% test accuracy with the 122-column autoencoder against 83.0% with a 40-column one). A fully connected autoencoder with encoder $f_\theta$ and decoder $g_\phi$ and layer sizes $122 \to 32 \to 8 \to 32 \to 122$ (ReLU activations, linear output) is trained to minimise the reconstruction loss

$$\mathcal{L}(\theta, \phi) = \frac{1}{|\mathcal{N}|} \sum_{x \in \mathcal{N}} \left\| x - g_\phi(f_\theta(x)) \right\|_2^2$$

with the Adam optimiser (learning rate $10^{-3}$, batch size 256, at most 60 epochs, early stopping on a 10% validation slice). Because the 8-dimensional bottleneck can only encode what is common to benign traffic, a record that does not lie on the benign manifold reconstructs poorly. The anomaly score of any record is

$$a(x) = \frac{1}{122} \left\| x - g_\phi(f_\theta(x)) \right\|_2^2,$$

and it is scaled to $[0, 1]$ by dividing by the 99.9th percentile of $a$ over the training split and clipping. With $z \in [0,1]^{40}$ the HRA-FS projection of $x$, the fused representation is $\hat{z} = [z, \; \tilde{a}(x)] \in [0,1]^{41}$. For reference, the autoencoder is also evaluated as a stand-alone detector with the decision rule $a(x) > \tau$, where $\tau$ is the 95th percentile of the benign training scores.

### 3.6 Novelty-aware stacked heterogeneous ensemble

Four level-0 learners are trained on $\hat{z}$: Random Forest (300 trees, balanced-subsample class weights), Extra Trees (300 trees, balanced class weights), XGBoost (400 rounds, depth 8, learning rate 0.08, histogram splits) and LightGBM (400 rounds, 63 leaves, learning rate 0.05, balanced class weights). Diversity comes from the different split-selection and boosting strategies: bagging with greedy splits, bagging with random splits, level-wise boosting and leaf-wise boosting.

Stacked generalisation proceeds as follows. The training set is partitioned into $K = 5$ folds. For each fold $k$ and each learner $m$, the learner is fitted on the other four folds and produces class probabilities $p^{(m)}_{i} \in \Delta^{C}$ for the records $i$ in fold $k$. Concatenating across learners and appending the anomaly score gives the out-of-fold meta-feature vector

$$u_i = \left[ p^{(1)}_i, \; p^{(2)}_i, \; p^{(3)}_i, \; p^{(4)}_i, \; \tilde{a}_i \right] \in \mathbb{R}^{4C + 1}.$$

Because $u_i$ was produced by models that never saw record $i$, it is an honest estimate of how each learner behaves on unseen data. The novelty of the proposed scheme lies in *how* the folds are drawn. With ordinary stratified folds, the base learners have seen every attack type during fitting and their out-of-fold probabilities on NSL-KDD are almost always correct and confident; a meta-learner trained on such rows learns to copy them and gives the anomaly score a negligible weight. We observed exactly this in a preliminary experiment: with stratified folds the meta-weight on the anomaly score was 0.98 and test accuracy did not move when the score was added. The dual-regime scheme therefore builds the meta-training set from two passes. In the *familiar* regime the folds are stratified by class. In the *novel* regime every attack type (neptune, smurf, warezclient, ...) is assigned to exactly one fold by a greedy balanced partition, while benign records are spread uniformly, so that when fold $k$ is held out the base learners have never seen the attack types it contains. Their out-of-fold probabilities for those records are what they would produce for a zero-day attack. The two passes give meta-sets $\{(u^{\text{fam}}_i, y_i)\}$ and $\{(u^{\text{nov}}_i, y_i)\}$ of $n$ rows each, which are concatenated into a $2n$-row meta-set. The level-1 model is a multinomial logistic regression with balanced class weights,

$$P(y = c \mid u) = \frac{\exp(w_c^\top u + b_c)}{\sum_{c'} \exp(w_{c'}^\top u + b_{c'})},$$

fitted on the $2n$-row meta-set. Seeing both regimes, the meta-learner learns a rule of the form "when the base learners agree and are confident, follow them; when they are unsure, let the anomaly score decide". Finally, each level-0 learner is refitted on the whole training set, and at inference the four full-data learners produce $u$ for a new record and the logistic layer gives the decision. The weights $w_c$ are reported in Section 4 because they show how much the system trusts each learner and the anomaly score. The same architecture is used for the binary task with $C = 2$ and for the five-class task with $C = 5$. Algorithm 2 gives the complete training procedure.

```
Algorithm 2: AESE-NIDS training
Input : KDDTrain+ records (X_raw, y_label), hyper-parameters k=40, K=5
Output: preprocessor P, mask M, autoencoder A, scale s, stacked model S
1  P <- fit log1p + min-max + one-hot on X_raw;  X <- P(X_raw)     # n x 122
2  M <- HRA-FS(X, y_5class, k);  Z <- X[:, M]                      # n x 40
3  A <- train autoencoder on X[y == normal] (Section 3.5, all 122 columns)
4  a <- A.reconstruction_error(X);  s <- quantile(a, 0.999)
5  Zhat <- [Z, clip(a / s, 0, 1)]                                   # n x 41
6  F_fam <- stratified K-fold ids of y
7  F_nov <- fold ids with every attack type in exactly one fold (greedy balanced), benign spread uniformly
8  for regime r in {fam, nov}:
9      for each level-0 learner m in {RF, ET, XGB, LGBM}:
10         U_r[:, m] <- out-of-fold predict_proba of m on (Zhat, y) using folds F_r
11     U_r <- [U_r, Zhat[:, -1]]                                    # append anomaly score
12 S.meta <- LogisticRegression(balanced).fit([U_fam; U_nov], [y; y])  # 2n rows
13 for each m: S.base[m] <- m.fit(Zhat, y)                          # refit on all data
14 return P, M, A, s, S
Inference: zhat <- [P(x)[:, M], clip(A.err(P(x)) / s, 0, 1)];  u <- [S.base[m].predict_proba(zhat), zhat[-1]];  y <- S.meta.predict(u)
```

### 3.7 Evaluation protocol and metrics

Models are trained on KDDTrain+ and evaluated once on KDDTest+. With TP, TN, FP and FN counted from the confusion matrix, the metrics are accuracy $(TP+TN)/(TP+TN+FP+FN)$, precision $TP/(TP+FP)$, recall or detection rate (DR) $TP/(TP+FN)$, F1 $= 2PR/(P+R)$, false-alarm rate (FAR) $FP/(FP+TN)$ and the Matthews correlation coefficient $\text{MCC} = (TP\cdot TN - FP\cdot FN)/\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}$. For the five-class task precision, recall and F1 are macro-averaged so that the rare R2L and U2R classes count as much as DoS. The area under the ROC curve is reported for the binary task. The unseen-attack detection rate is the fraction of the test records whose attack label does not occur in training that the binary model flags as attacks. Five-fold stratified cross-validation on KDDTrain+ is reported as well, to show the gap between in-distribution and shifted evaluation. Training time and per-record inference time are measured on the hardware listed in Section 4.1.

### 3.8 Implementation

The system is implemented in Python 3.12 with scikit-learn 1.9 [21], XGBoost 3.4, LightGBM 4.7, NumPy, pandas, matplotlib and seaborn. The code is organised as a small package in `src/`: `preprocessing.py` builds the column transformer, `feature_selection.py` implements Algorithm 1, `autoencoder.py` wraps the bottleneck network, `models.py` defines the baselines and the stacked ensemble, `train.py` runs the complete experiment and writes every table and figure, `compare.py` produces the comparison with published results and `predict.py` scores new records with the saved pipeline. The full listing is given in Section 7.

---

{{RESULTS}}
