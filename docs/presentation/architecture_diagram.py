"""Editorial pipeline diagram of the NIDS, 1920 x 1080. Font: Inter (Rasmus Andersson), Segoe UI fallback.
Writes architecture_diagram.png and architecture_diagram.svg next to this file.
Run: python docs/presentation/architecture_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent
# palette of the presentation deck: white, periwinkle 6C78E6 and its tint EEF0FC, black text, grey captions
BG, CARD, STONE, INK, MUTED = "#FFFFFF", "#FFFFFF", "#C9CEF5", "#111111", "#555555"
PERI, TINT = "#6C78E6", "#EEF0FC"
ACC = {  # fill, ink (text and border)
    "data": (TINT, "#111111"),
    "feat": ("#FFFFFF", PERI),
    "model": (PERI, "#FFFFFF"),
    "stack": (TINT, "#111111"),
    "out": ("#FFFFFF", "#111111"),
}
FONT = next((f for f in ("Inter", "Segoe UI", "Helvetica Neue", "Arial") if any(x.name == f for x in font_manager.fontManager.ttflist)), "DejaVu Sans")
plt.rcParams["font.family"] = FONT
plt.rcParams["svg.fonttype"] = "none"

W, H = 1920, 1080
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

PX = 72 / 100   # 1 px = 0.72 pt at dpi 100


def pt(px): return px * PX


def rbox(x, y, w, h, fill, edge, lw=1.0, r=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill, ec=edge, lw=lw))


def node(x, y, w, h, title, sub, key, title_px=15, sub_px=11.5):
    fill, ink = ACC[key]
    rbox(x, y, w, h, fill, PERI if ink == "#111111" and fill != "#FFFFFF" else ink, lw=1.2)
    lines_sub = sub.split("\n") if sub else []
    th = title_px * 1.25; sh = sub_px * 1.35
    total = th + (len(lines_sub) * sh + 6 if lines_sub else 0)
    y0 = y + h / 2 - total / 2 + th * 0.78
    ax.text(x + w / 2, y0, title, ha="center", va="baseline", fontsize=pt(title_px), fontweight="bold", color=ink)
    for i, ln in enumerate(lines_sub):
        ax.text(x + w / 2, y0 + 8 + sh * (i + 1), ln, ha="center", va="baseline", fontsize=pt(sub_px), color=ink)


def label(x, y, text, key, px=12.5):
    ax.text(x, y, text, ha="left", va="baseline", fontsize=pt(px), fontweight="bold", color=PERI)


def container(x, y, w, h):
    rbox(x, y, w, h, CARD, STONE, lw=1.0, r=12)


def arrow(points, color=INK, head=True):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=1.4, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    if head:
        (x1, y1), (x2, y2) = points[-2], points[-1]
        dx, dy = x2 - x1, y2 - y1
        n = (dx ** 2 + dy ** 2) ** 0.5; dx, dy = dx / n, dy / n
        s = 9
        ax.fill([x2, x2 - s * dx + s * 0.55 * dy, x2 - s * dx - s * 0.55 * dy], [y2, y2 - s * dy - s * 0.55 * dx, y2 - s * dy + s * 0.55 * dx], color=color, zorder=4)


def small(x, y, text, color=MUTED, px=10.5, ha="center", italic=False):
    ax.text(x, y, text, ha=ha, va="baseline", fontsize=pt(px), color=color, style="italic" if italic else "normal")


# ---------------------------------------------------------------- header
ax.text(60, 78, "Intelligent Network Intrusion Detection System", ha="left", va="baseline", fontsize=pt(30), fontweight="bold", color=INK)
ax.text(60, 108, "Novelty-aware dual-regime stacking architecture on NSL-KDD", ha="left", va="baseline", fontsize=pt(15), color=MUTED)
ax.plot([60, 1860], [136, 136], color=STONE, lw=1)

# ---------------------------------------------------------------- columns
cols = {"c1": 60, "c2": 430, "c3": 800, "c4": 1236, "c5": 1596}
CW, CW5 = 300, 280
TOP, BOT = 170, 870

# column 1: data and preprocessing
container(cols["c1"] - 16, TOP, CW + 32, 420)
label(cols["c1"], TOP + 32, "01   DATA AND PREPROCESSING", "data")
node(cols["c1"], 230, CW, 112, "NSL-KDD Dataset", "KDDTrain+ 125,973 records\nKDDTest+ 22,544 records\n41 raw features", "data")
arrow([(cols["c1"] + CW / 2, 342), (cols["c1"] + CW / 2, 392)])
node(cols["c1"], 392, CW, 112, "Preprocessing", "log1p on heavy-tailed counters\nmin-max normalization\none-hot encoding, 122 columns", "data")
arrow([(cols["c1"] + CW, 448), (cols["c2"] - 16, 448)])

# column 2: feature selection
container(cols["c2"] - 16, TOP, CW + 32, 420)
label(cols["c2"], TOP + 32, "02   FEATURE SELECTION", "feat")
node(cols["c2"], 230, CW, 112, "Multi-Criteria Ranking", "mutual information\nANOVA F-test\nExtra-Trees importance", "feat")
arrow([(cols["c2"] + CW / 2, 342), (cols["c2"] + CW / 2, 392)])
node(cols["c2"], 392, CW, 112, "Borda-Count Aggregation", "ranks summed across the three views\nconsensus top-40 feature subset", "feat")

GAP_A, GAP_B = 758, 772           # two lanes in the gap between columns 2 and 3
# selected features -> supervised learners (track C, top of column 3)
arrow([(cols["c2"] + CW, 448), (GAP_B, 448), (GAP_B, 276), (cols["c3"] - 16, 276)])
# all 122 columns, benign rows only -> both autoencoder tracks
arrow([(cols["c1"] + CW, 480), (cols["c1"] + CW + 30, 480), (cols["c1"] + CW + 30, 620), (GAP_A, 620), (GAP_A, 528), (cols["c3"] - 16, 528)])
arrow([(GAP_A, 620), (GAP_A, 770), (cols["c3"] - 16, 770)])
small(cols["c1"] + CW + 44, 612, "all 122 columns, benign records only", ha="left", italic=True)

# column 3: parallel base learners
C3W = 370
container(cols["c3"] - 16, TOP, C3W + 32, BOT - TOP)
label(cols["c3"], TOP + 32, "03   PARALLEL BASE LEARNERS", "model")
small(cols["c3"], TOP + 54, "Track C, supervised, 40 selected features", ha="left", px=11)
node(cols["c3"], 230, C3W, 92, "Supervised Base Learners", "Random Forest, Extra Trees, XGBoost, LightGBM\nclass-balanced, 300 trees / 400 rounds", "model")
arrow([(cols["c3"] + C3W / 2, 322), (cols["c3"] + C3W / 2, 352)])
node(cols["c3"] + 80, 352, 210, 56, "Class Probability Vectors", "out-of-fold predictions", "model", title_px=12.5, sub_px=10)

small(cols["c3"], 470, "Track A, unsupervised, benign-only", ha="left", px=11)
node(cols["c3"], 486, 170, 84, "Feature Clustering", "correlation distance\ngroups of at most 10", "model", title_px=13, sub_px=10.5)
arrow([(cols["c3"] + 170, 528), (cols["c3"] + 200, 528)])
node(cols["c3"] + 200, 486, 170, 84, "31 Subspace Autoencoders", "(m = 10), one per group\nbenign traffic only", "model", title_px=12, sub_px=10.5)
arrow([(cols["c3"] + 285, 570), (cols["c3"] + 285, 600)])
node(cols["c3"] + 160, 600, 210, 56, "Subspace RMSE Vector", "31 anomaly scores", "model", title_px=12.5, sub_px=10)

small(cols["c3"], 712, "Track B, unsupervised, benign-only", ha="left", px=11)
node(cols["c3"], 728, 170, 84, "Standalone Autoencoder", "122-32-8-32-122\nbenign traffic only", "model", title_px=12.5, sub_px=10.5)
arrow([(cols["c3"] + 170, 770), (cols["c3"] + 200, 770)])
node(cols["c3"] + 200, 728, 170, 84, "Reconstruction Error", "single anomaly score", "model", title_px=12.5, sub_px=10.5)

# column 4: novelty-aware dual-regime stacking
C4W = 320
container(cols["c4"] - 16, TOP, C4W + 32, BOT - TOP)
label(cols["c4"], TOP + 32, "04   NOVELTY-AWARE DUAL-REGIME STACKING", "stack", px=12)
node(cols["c4"], 230, C4W, 96, "Meta-Training Set Construction", "out-of-fold predictions generated twice,\nonce under each fold regime", "stack", title_px=14)
arrow([(cols["c4"] + 75, 326), (cols["c4"] + 75, 372)])
arrow([(cols["c4"] + C4W - 75, 326), (cols["c4"] + C4W - 75, 372)])
node(cols["c4"], 372, 150, 150, "Familiar Regime", "stratified 5-fold splits\nlearners have seen\nevery attack type", "stack", title_px=12.5, sub_px=10)
node(cols["c4"] + C4W - 150, 372, 150, 150, "Novel Regime", "attack-type folds,\neach type held out once\nlearners are fooled", "stack", title_px=12.5, sub_px=10)
arrow([(cols["c4"] + 75, 522), (cols["c4"] + 75, 560), (cols["c4"] + C4W / 2, 560), (cols["c4"] + C4W / 2, 596)])
arrow([(cols["c4"] + C4W - 75, 522), (cols["c4"] + C4W - 75, 560), (cols["c4"] + C4W / 2, 560)], head=False)
node(cols["c4"], 596, C4W, 110, "Logistic Meta-Learner", "inputs: class probabilities\n+ subspace RMSE vector (binary task)\n+ reconstruction error (5-class task)", "stack", title_px=14)
small(cols["c4"] + C4W / 2, 744, "learns when to follow the base learners", px=10.5)
small(cols["c4"] + C4W / 2, 760, "and when to defer to the anomaly features", px=10.5)
small(cols["c4"] + C4W / 2, 804, "weight on the anomaly features:", px=10.5)
small(cols["c4"] + C4W / 2, 820, "0.98 with one regime, 5.3 with two", px=10.5)

# column 3 -> column 4 feeds, three lanes in the gap
L1, L2, L3 = 1194, 1204, 1214
arrow([(cols["c3"] + 290, 380), (L1, 380), (L1, 278), (cols["c4"] - 16, 278)])      # probabilities -> meta-set
arrow([(cols["c3"] + 370, 628), (L2, 628), (L2, 640), (cols["c4"] - 16, 640)])      # subspace vector -> meta-learner
arrow([(cols["c3"] + 370, 770), (L3, 770), (L3, 662), (cols["c4"] - 16, 662)])      # reconstruction error -> meta-learner

# column 5: classification and evaluation
container(cols["c5"] - 16, TOP, CW5 + 32, 500)
label(cols["c5"], TOP + 32, "05   CLASSIFICATION AND EVALUATION", "out", px=11.5)
node(cols["c5"], 230, CW5, 112, "Final Traffic Classification", "binary: Normal | Attack\n5-class: Normal | DoS | Probe | R2L | U2R", "out", title_px=14, sub_px=11)
arrow([(cols["c4"] + C4W, 651), (cols["c5"] - 16 - 16, 651), (cols["c5"] - 16 - 16, 286), (cols["c5"] - 16, 286)])
arrow([(cols["c5"] + CW5 / 2, 342), (cols["c5"] + CW5 / 2, 392)])
node(cols["c5"], 392, CW5, 128, "Evaluation Suite", "accuracy, macro F1-score, ROC-AUC\nfalse-alarm rate, unseen-attack detection\nKDDTest-21, fixed false-alarm budgets", "out", title_px=14, sub_px=10.5)
small(cols["c5"] + CW5 / 2, 548, "86.5% accuracy  |  79.6% of unseen attack types", color=PERI, px=11)

# footer
small(60, 1040, "Every stage is fitted on KDDTrain+ only. KDDTest+ is used once, for the final evaluation.", ha="left", px=11)

fig.savefig(OUT / "architecture_diagram.png", dpi=100, facecolor=BG)
fig.savefig(OUT / "architecture_diagram.svg", facecolor=BG)
print("font:", FONT, "| wrote architecture_diagram.png and .svg")
