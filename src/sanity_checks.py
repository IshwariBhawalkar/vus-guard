"""Sanity checks that guard against fooling ourselves.

1. Label-permutation test: shuffle labels inside the training data. If the pipeline leaks,
   AUC stays high. It should collapse to ~0.5.
2. Leave-one-gene-out (LOGO) on missense variants: test on genes the model has never seen.
   This is the harder, more honest generalisation test than position-held-out CV.

Run:  python3 src/sanity_checks.py
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (ROOT, N_FOLDS, group_folds, load_data, make_model, make_X)  # noqa: E402


def cv_auc(lab, seed, shuffle_labels=False, group_by="position", missense_only=True):
    rng = np.random.RandomState(seed)
    y = (lab["label"] == "R").astype(int).values
    groups = lab["group"].values if group_by == "position" else lab["gene"].values
    fold = group_folds(groups, seed, N_FOLDS)
    p = np.full(len(lab), np.nan)
    for f in range(N_FOLDS):
        te = fold == f
        tr = lab[~te].copy()
        if shuffle_labels:
            tr["label"] = rng.permutation(tr["label"].values)
        Xtr, Xte = make_X(tr, tr), make_X(lab[te], tr)
        m = make_model("logreg", seed).fit(Xtr, (tr["label"] == "R").astype(int).values)
        p[te] = m.predict_proba(Xte)[:, 1]
    msk = (lab["effect"] == "missense_variant").values if missense_only else np.ones(len(lab), bool)
    return float(roc_auc_score(y[msk], p[msk]))


def main():
    base = load_data()
    lab = base[base["label"].isin(["R", "S"])].reset_index(drop=True)
    res = {}
    res["position_held_out_missense_auc"] = [cv_auc(lab, s) for s in range(3)]
    res["label_permutation_missense_auc"] = [cv_auc(lab, s, shuffle_labels=True) for s in range(3)]
    res["leave_gene_out_missense_auc"] = [cv_auc(lab, s, group_by="gene") for s in range(3)]
    for k, v in res.items():
        print(f"{k:38s} mean {np.mean(v):.3f}  runs {np.round(v, 3).tolist()}")
    with open(os.path.join(ROOT, "reports", "sanity_checks.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
