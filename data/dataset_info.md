# The dataset
---
## 1. The 41 features

**Layer 1.** Think of a connection as a phone call. Basic features are the call log
(who, how long, which line). Content features are what was said (did someone ask for
the admin password?). Traffic features are the pattern of calls around it (how many
calls to the same number in the last two seconds?).

**Layer 2.** Four groups:

| Group | Count | Examples | Typical signal |
|---|---|---|---|
| Basic | 9 | `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent` | protocol anomalies, size anomalies |
| Content | 13 | `hot`, `num_failed_logins`, `logged_in`, `num_compromised`, `root_shell`, `su_attempted`, `num_root`, `num_file_creations`, `num_shells`, `num_access_files`, `is_guest_login` ... | R2L and U2R (payload-level events) |
| Time-based traffic (2-second window) | 9 | `count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `same_srv_rate`, `diff_srv_rate` ... | DoS floods, scans |
| Host-based traffic (last 100 connections to the host) | 10 | `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`, `dst_host_serror_rate` ... | slow scans, distributed floods |

Three features are categorical strings: `protocol_type` (tcp, udp, icmp), `service`
(70 values: http, smtp, ftp_data, ...), `flag` (11 TCP state flags: SF normal, S0 half-open,
REJ rejected, ...). After one-hot encoding the record has **122 numeric columns**
(38 numeric + 84 indicator columns).

Scale problems you must know: `src_bytes` ranges from 0 to about 1.4 billion. Sixteen
counters like this are heavy-tailed, which is why the pipeline applies `log(1 + x)`
before scaling (`src/config.py::LOG_FEATURES`).

**Layer 3.** Why does the feature group matter? Because each attack family lives in a
different group. A SYN flood (DoS) shows up as high `count` and high `serror_rate`
(many half-open connections, flag S0). A port scan (Probe) shows up as many REJ flags
and low `same_srv_rate`. A password-guessing attack (R2L) shows up only in
`num_failed_logins` and looks perfectly ordinary in the traffic features. This is the
reason the v2 model gives the classifier a *per-group* anomaly profile instead of one
number (see notes 02 and 03).

---

## 2. Labels: 39 attack names, 4 families, 2 tasks

**Layer 1.** Every row is either "normal" or one of 39 named attacks. We group the
attack names into four families, the way a hospital groups hundreds of diagnoses into
departments.

**Layer 2.** The mapping is `src/config.py::ATTACK_CATEGORY`:

| Family | Meaning | Examples (train) | Train rows | Test rows |
|---|---|---|---|---|
| Normal | benign | - | 67,343 | 9,711 |
| DoS | Denial of Service: exhaust a resource | neptune (SYN flood), smurf (ICMP amplification), back, teardrop, pod, land | 45,927 | 7,460 |
| Probe | reconnaissance | satan, ipsweep, portsweep, nmap | 11,656 | 2,421 |
| R2L | Remote-to-Local: unauthorised access from outside | guess_passwd, warezclient, warezmaster, ftp_write, imap, phf | 995 | 2,885 |
| U2R | User-to-Root: privilege escalation | buffer_overflow, rootkit, loadmodule, perl | 52 | 67 |

Two tasks are solved from the same features:

- **Binary**: `is_attack` = 1 if label != "normal". Metrics: accuracy, precision,
  recall (= detection rate), F1, MCC, false-alarm rate.
- **5-class**: `category` in {Normal, DoS, Probe, R2L, U2R}. Metrics are macro-averaged
  so the 67-row U2R class counts as much as the 9,711-row Normal class.

**Layer 3.** Three things to say about this table in an interview:

1. *Imbalance.* U2R has 52 training rows. Without class weighting, every model predicts
   it never. All tree learners in `../src/models.py` use `class_weight="balanced"` or
   `balanced_subsample`; the logistic meta-learner too.
2. *Shift in R2L.* R2L is 0.8% of training and 12.8% of test. The test set deliberately
   over-represents the family that is hardest to learn.
3. *Why macro averages.* With accuracy alone, a model that ignores R2L and U2R entirely
   still scores about 79%. Macro-F1 exposes that.

---

## 3. The trap: 17 attack types exist only in the test set

**Layer 1.** Imagine studying for an exam from last year's paper, and this year's paper
has new question types nobody showed you. NSL-KDD's test set does this on purpose.

**Layer 2.** `../src/data_loader.py` prints them:

```
apache2, httptunnel, mailbomb, mscan, named, processtable, ps, saint, sendmail,
snmpgetattack, snmpguess, sqlattack, udpstorm, worm, xlock, xsnoop, xterm
```

They account for **3,750 of the 22,544 test rows (16.6%)**. Most are R2L
(snmpgetattack, snmpguess, httptunnel, named, sendmail, xlock, xsnoop), which is why
R2L is the hardest family. `snmpgetattack` alone is 178 rows of SNMP traffic that is
feature-for-feature nearly identical to benign SNMP traffic.

The project reports a dedicated metric, **unseen-attack detection rate** = fraction of
those 3,750 rows that the binary model flags. It is computed in `../src/train.py`
(`unseen_mask`) and saved to `results/tables/unseen_attack_detection.csv`.

**Layer 3.** This is the single most important number in the project, and it is the
one that separates methods:

| Model (same 71-column input) | Unseen-attack detection |
|---|---|
| Random Forest | 26.4% |
| XGBoost | 41.2% |
| Proposed v1 (single anomaly score) | 64.9% |
| Proposed v2 (subspace profile) | 79.6% |

Supervised models can only learn boundaries around attack types they have seen.
Catching the other 17 requires a signal that says "this does not look like normal",
which is what the benign-only autoencoders provide.

---

## 4. The `difficulty` column and KDDTest-21

**Layer 1.** When Tavallaee built NSL-KDD they asked 21 different "students" (learners)
to classify every record and wrote down how many got it right. A record that all 21
got right is easy; one that only 3 got right is hard. That count is stored with every
row.

**Layer 2.** The 43rd field, `difficulty`, is an integer 0-21: the number of the 21
reference learners (7 algorithms x 3 seeds: J48, Naive Bayes, NB-Tree, Random Forest,
Random Tree, MLP, SVM) that classified the record correctly. NSL-KDD's test set was
sampled *inversely* to this count, so hard records are over-represented.

**KDDTest-21** = the test rows with `difficulty < 21`, i.e. records that at least one
reference learner got wrong. In our split that is **11,850 rows (52.6% of KDDTest+)**.
`../experiments/run_experiments.py` and `src/train.py` both compute it:

```python
test21 = (test_df["difficulty"] < 21).to_numpy()
```

Our numbers on it (`../results/tables/results_test21.csv`):

| Model | Binary acc | Detection | False alarms |
|---|---|---|---|
| Random Forest | 56.7 | 49.8 | 12.2 |
| XGBoost | 61.6 | 55.9 | 12.5 |
| Proposed v2 | 74.6 | 77.5 | 38.7 |

**Layer 3.** Two lessons. First, the 85% ceiling everyone hits on KDDTest+ is by
construction: half the test rows were chosen because 2009-era learners failed on them.
Second, our high false-alarm rate on Test-21 (38.7%) is the honest cost of anomaly
detection: the benign rows in the hard subset are benign connections that *look*
unusual, and an anomaly score flags unusual things. In an interview, say this before
the interviewer finds it.

---