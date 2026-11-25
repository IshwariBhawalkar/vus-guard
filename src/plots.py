"""Make the three report figures from reports/*.json|csv.  Run: python3 src/plots.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "reports", *p)

BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"   # validated categorical slots 1-3
GRAY, INK, MUTED, GRID = "#9c9c9c", "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})

m = json.load(open(R("metrics.json")))
s = json.load(open(R("sanity_checks.json")))
best, alpha = m["selected_model"], str(m["alpha"])


def fig1_validation():
    mm = m["summary"]["missense_only"]["models"]
    rows = [("Label-shuffled (should be ~0.5)", sum(s["label_permutation_missense_auc"]) / 3, GRAY),
            ("Unseen GENES (leave-gene-out)", sum(s["leave_gene_out_missense_auc"]) / 3, GRAY),
            ("Baseline: gene prior only", mm["gene_prior"]["roc_auc"]["mean"], GRAY),
            ("Baseline: hotspot distance only", mm["hotspot_only"]["roc_auc"]["mean"], GRAY),
            ("Random forest", mm["random_forest"]["roc_auc"]["mean"], BLUE),
            ("Logistic regression (selected)", mm["logreg"]["roc_auc"]["mean"], BLUE)]
    fig, ax = plt.subplots(figsize=(8, 3.9), dpi=200)
    for i, (name, v, c) in enumerate(rows):
        ax.barh(i, v, color=c, height=0.62)
        ax.text(v + 0.012, i, f"{v:.2f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.axvline(0.5, color=GRID, lw=1, zorder=0)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("ROC-AUC on missense mutations (630 labelled; unseen positions unless stated)")
    ax.set_title("Good on new positions in known genes; not on new genes",
                 fontsize=11, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(R("fig1_validation.png"))


def fig2_abstention():
    c = m["conformal"][best]["missense_only"]
    al = sorted(float(a) for a in c)
    ab = [100 * c[str(a)]["abstain_rate"] for a in al]
    acc = [100 * c[str(a)]["acc_decided"] for a in al]
    base = 100 * m["summary"]["missense_only"]["models"][best]["bal_acc_no_abstain"]["mean"]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    ax.scatter(ab, acc, s=70, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
    for a, x, y in zip(al, ab, acc):
        ax.annotate(f"α={a:g}", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8.5, color=MUTED)
    ax.axhline(base, color=ORANGE, lw=1.5, ls=(0, (4, 3)))
    ax.text(max(ab) + 3, base + 0.25, "No abstention (p ≥ 0.5 rule)", color=ORANGE, fontsize=9, ha="right")
    ax.set_ylim(89, 97.5)
    ax.set_xlabel("Share of mutations the model abstains on (%)")
    ax.set_ylabel("Balanced accuracy on mutations it answers (%)")
    ax.set_title("Accuracy on answered mutations vs share abstained (missense)",
                 fontsize=11, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(R("fig2_abstention.png"))


def fig3_vus():
    p = pd.read_csv(R("vus_predictions.csv"))
    t = p.groupby(["drug", "decision"]).size().unstack(fill_value=0)
    t = t.reindex(columns=["R", "abstain", "S"], fill_value=0)
    t["tot"] = t.sum(axis=1)
    t = t.sort_values("tot")
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=200)
    left = None
    labels = {"R": ("Predicted resistant", BLUE), "abstain": ("Abstain → lab test", ORANGE),
              "S": ("Predicted susceptible", AQUA)}
    import numpy as np
    left = np.zeros(len(t))
    for k in ["R", "abstain", "S"]:
        share = 100 * t[k].values / t["tot"].values
        ax.barh(range(len(t)), share, left=left, color=labels[k][1], height=0.7,
                edgecolor="white", linewidth=1.5, label=labels[k][0])
        left += share
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"{d} (n={int(n):,})" for d, n in zip(t.index, t["tot"])], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of the drug's uncertain-significance mutations")
    ax.set_title("Model output on the WHO grade-3 mutations (hypotheses, not validated calls)",
                 fontsize=11, loc="left", color=INK)
    ax.legend(ncol=3, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.45, -0.12))
    fig.tight_layout()
    fig.savefig(R("fig3_vus_predictions.png"))


if __name__ == "__main__":
    fig1_validation()
    fig2_abstention()
    fig3_vus()
    print("figures written to reports/")
