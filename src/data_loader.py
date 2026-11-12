"""Loading NSL-KDD and attaching binary / 5-class targets."""
import pandas as pd

from config import COLUMNS, ATTACK_CATEGORY, TRAIN_FILE, TEST_FILE


def load_split(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["category"] = df["label"].map(ATTACK_CATEGORY).fillna("R2L")  # unseen rare names default to R2L
    df["is_attack"] = (df["label"] != "normal").astype(int)
    return df


def load_nsl_kdd():
    train = load_split(TRAIN_FILE)
    test = load_split(TEST_FILE)
    return train, test


if __name__ == "__main__":
    tr, te = load_nsl_kdd()
    print("Train:", tr.shape, "Test:", te.shape)
    print("\nTrain category distribution:\n", tr["category"].value_counts())
    print("\nTest category distribution:\n", te["category"].value_counts())
    unseen = set(te["label"]) - set(tr["label"])
    print("\nAttack types present only in the test set:", sorted(unseen))
