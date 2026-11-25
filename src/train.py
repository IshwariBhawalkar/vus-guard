"""VUS-Guard training + evaluation.

Pipeline
  1. Load WHO catalogue (slim CSV), keep in-scope effects, label R (grades 1-2) / S (4-5);
     grade 3 (uncertain) is the prediction target and has no ground truth.
  2. Features: sequence-derived only (see features.py).
  3. Evaluation: repeated GROUPED 5-fold CV. Rows sharing a gene+position never straddle
     train and test, so the model is always tested on positions it has not seen.
  4. Uncertainty: class-conditional (Mondrian) split-conformal prediction. The model
     outputs {R}, {S}, or abstains when the prediction set is ambiguous or empty.
  5. Final model + predictions for all uncertain variants.

Run:  python3 src/train.py
"""
import json
import logging
import math
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (CAT_COLS, IN_SCOPE_EFFECTS, NUM_COLS, build_base_table,  # noqa: E402
                      neighbour_features)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed", "catalogue_slim.csv")
SEEDS = [0, 1, 2]          # outer-CV repeats
N_FOLDS = 5
CAL_FRAC = 0.25            # share of training groups held out for conformal calibration
ALPHA = 0.10               # target miscoverage (90% coverage per class and bin)
ALPHAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
FINAL_SEED = 42

os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
log = logging.getLogger("vus-guard")


def setup_logging():
    """Configured only when train.py is RUN (not imported), so importing never wipes the log."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(ROOT, "logs", "training_log.txt"), "w"),
                                  logging.StreamHandler()])


# ----------------------------------------------------------------------------- data
def load_data():
    df = pd.read_csv(DATA)
    df["lab"] = df["final_grade"].str[0].map({"1": "R", "2": "R", "4": "S", "5": "S", "3": "U"})
    df = df[df["effect"].isin(IN_SCOPE_EFFECTS)].reset_index(drop=True)
    base = build_base_table(df)
    return base


def make_X(target, ref):
    """Intrinsic features + hotspot features computed against `ref` labelled rows."""
    nb = neighbour_features(target, ref)
    X = pd.concat([target[CAT_COLS[:3] + ["ctype"]], target[[c for c in NUM_COLS if c not in nb.columns]], nb], axis=1)
    return X[CAT_COLS + NUM_COLS]


# --------------------------------------------------------------------------- models
def preprocessor():
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
                              ("num", StandardScaler(), NUM_COLS)])


def make_model(name, seed):
    if name == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                     class_weight="balanced_subsample", random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("prep", preprocessor()), ("clf", clf)])


def gene_prior_scores(train, test, m=2.0):
    """Baseline: per (drug, gene) smoothed fraction of resistant rows. No sequence features."""
    glob = (train["label"] == "R").mean()
    g = train.groupby(["drug", "gene"])["label"].agg(lambda s: ((s == "R").sum() + m * glob) / (len(s) + m))
    return np.array([g.get((d, ge), glob) for d, ge in zip(test["drug"], test["gene"])])


def hotspot_only_scores(Xtr, ytr, Xte):
    """Baseline: logistic regression on the four hotspot-distance features only."""
    cols = ["dR", "dS", "nR10", "nS10"]
    m = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(class_weight="balanced", max_iter=2000))])
    m.fit(Xtr[cols], ytr)
    return m.predict_proba(Xte[cols])[:, 1]


# ------------------------------------------------------------------------- conformal
def bin_of(effect):
    return np.where(np.asarray(effect) == "missense_variant", "missense", "other")


def conformal_thresholds(p_cal, y_cal, bins_cal, alpha):
    """Mondrian split conformal: one threshold per (true class, variant-bin).
    Nonconformity score = 1 - P(true class). A class enters the prediction set when
    1 - P(class) <= its threshold. Too few calibration points -> threshold = inf (always
    included), which only makes the model abstain more, never less safe."""
    q = {}
    for cls in (0, 1):
        for b in ("missense", "other"):
            mask = (y_cal == cls) & (bins_cal == b)
            n = int(mask.sum())
            score = (1 - p_cal[mask]) if cls == 1 else p_cal[mask]
            level = math.ceil((n + 1) * (1 - alpha)) / n if n > 0 else 2.0
            q[(cls, b)] = float(np.quantile(score, level, method="higher")) if level <= 1 else float("inf")
    return q


def conformal_decide(p, bins, q):
    """Return decisions: 'R', 'S' or 'abstain', plus a boolean 'true class in set' helper."""
    in_R = (1 - p) <= np.array([q[(1, b)] for b in bins])
    in_S = p <= np.array([q[(0, b)] for b in bins])
    dec = np.where(in_R & ~in_S, "R", np.where(in_S & ~in_R, "S", "abstain"))
    return dec, in_R, in_S


def metrics(y, p, dec, in_R, in_S):
    out = {"n": int(len(y))}
    if len(set(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    pred05 = (p >= 0.5).astype(int)
    tpr = ((pred05 == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    tnr = ((pred05 == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    out["bal_acc_no_abstain"] = float((tpr + tnr) / 2)
    if dec is not None:
        covered = np.where(y == 1, in_R, in_S)
        out["coverage"] = float(covered.mean())
        out["abstain_rate"] = float((dec == "abstain").mean())
        d = dec != "abstain"
        out["n_decided"] = int(d.sum())
        if d.sum():
            yd, pd_ = y[d], (dec[d] == "R").astype(int)
            out["acc_decided"] = float((yd == pd_).mean())
            out["sens_decided"] = float(((pd_ == 1) & (yd == 1)).sum() / max((yd == 1).sum(), 1))
            out["spec_decided"] = float(((pd_ == 0) & (yd == 0)).sum() / max((yd == 0).sum(), 1))
            tprd = out["sens_decided"] if (yd == 1).sum() else np.nan
            tnrd = out["spec_decided"] if (yd == 0).sum() else np.nan
            out["bal_acc_decided"] = float(np.nanmean([tprd, tnrd]))
    return out


# ------------------------------------------------------------------------------ CV
def group_folds(groups, seed, n_folds):
    uniq = np.array(sorted(set(groups)))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def split_groups(groups, frac, seed):
    uniq = np.array(sorted(set(groups)))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    cal = set(uniq[: int(round(frac * len(uniq)))])
    return np.array([g in cal for g in groups])


def run_cv(lab):
    y_all = (lab["label"] == "R").astype(int).values
    model_names = ["logreg", "random_forest"]
    records = []                       # one entry per seed: dict of pooled predictions
    for seed in SEEDS:
        fold = group_folds(lab["group"].values, seed, N_FOLDS)
        pooled = {k: {"p": np.zeros(len(lab)), "dec": {a: np.empty(len(lab), dtype=object) for a in ALPHAS},
                      "inR": {a: np.zeros(len(lab), bool) for a in ALPHAS},
                      "inS": {a: np.zeros(len(lab), bool) for a in ALPHAS}}
                  for k in model_names + ["gene_prior", "hotspot_only"]}
        for f in range(N_FOLDS):
            te_mask = fold == f
            tr_all = lab[~te_mask]
            cal_mask = split_groups(tr_all["group"].values, CAL_FRAC, seed * 100 + f)
            proper, cal, test = tr_all[~cal_mask], tr_all[cal_mask], lab[te_mask]
            ref = proper
            Xp = make_X(proper, ref)
            Xc = make_X(cal, ref)
            Xt = make_X(test, ref)
            yp = (proper["label"] == "R").astype(int).values
            yc = (cal["label"] == "R").astype(int).values
            te_idx = np.where(te_mask)[0]
            bins_c, bins_t = bin_of(cal["effect"]), bin_of(test["effect"])
            for name in model_names:
                m = make_model(name, seed).fit(Xp, yp)
                pc, pt = m.predict_proba(Xc)[:, 1], m.predict_proba(Xt)[:, 1]
                pooled[name]["p"][te_idx] = pt
                for a in ALPHAS:
                    q = conformal_thresholds(pc, yc, bins_c, a)
                    dec, iR, iS = conformal_decide(pt, bins_t, q)
                    pooled[name]["dec"][a][te_idx], pooled[name]["inR"][a][te_idx], pooled[name]["inS"][a][te_idx] = dec, iR, iS
            pooled["gene_prior"]["p"][te_idx] = gene_prior_scores(proper, test)
            pooled["hotspot_only"]["p"][te_idx] = hotspot_only_scores(Xp, yp, Xt)
        records.append(pooled)
        log.info("seed %d done", seed)
    return y_all, records, model_names


def summarise(y_all, records, lab, model_names):
    is_mis = (lab["effect"] == "missense_variant").values
    subsets = {"missense_only": is_mis, "all_in_scope": np.ones(len(lab), bool)}
    summary = {}
    for sub_name, msk in subsets.items():
        summary[sub_name] = {"n": int(msk.sum()), "n_R": int(y_all[msk].sum()), "models": {}}
        for k in model_names + ["gene_prior", "hotspot_only"]:
            per_seed = []
            for rec in records:
                per_seed.append(metrics(y_all[msk], rec[k]["p"][msk], None, None, None))
            agg = {}
            for key in per_seed[0]:
                vals = [s[key] for s in per_seed]
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            summary[sub_name]["models"][k] = agg
    # conformal summaries for the model families that have them
    conf = {}
    for k in model_names:
        conf[k] = {}
        for sub_name, msk in subsets.items():
            conf[k][sub_name] = {}
            for a in ALPHAS:
                per_seed = [metrics(y_all[msk], rec[k]["p"][msk], rec[k]["dec"][a][msk],
                                    rec[k]["inR"][a][msk], rec[k]["inS"][a][msk]) for rec in records]
                conf[k][sub_name][str(a)] = {key: float(np.nanmean([s[key] for s in per_seed]))
                                             for key in per_seed[0] if key != "n"}
    return summary, conf


# ---------------------------------------------------------------------------- main
def main():
    setup_logging()
    t0 = time.time()
    import sklearn
    log.info('python %s | sklearn %s | pandas %s | numpy %s', sys.version.split()[0], sklearn.__version__, pd.__version__, np.__version__)
    base = load_data()
    lab = base[base["label"].isin(["R", "S"])].reset_index(drop=True)
    unc = base[base["label"] == "U"].reset_index(drop=True)
    log.info("in-scope rows: %d labelled (R=%d, S=%d), %d uncertain", len(lab),
             (lab.label == "R").sum(), (lab.label == "S").sum(), len(unc))
    log.info("missense labelled: R=%d S=%d", ((lab.label == "R") & (lab.effect == "missense_variant")).sum(),
             ((lab.label == "S") & (lab.effect == "missense_variant")).sum())

    y_all, records, model_names = run_cv(lab)
    summary, conf = summarise(y_all, records, lab, model_names)

    for sub in summary:
        log.info("=== %s (n=%d, R=%d) ===", sub, summary[sub]["n"], summary[sub]["n_R"])
        for k, v in summary[sub]["models"].items():
            log.info("%-14s ROC-AUC %.3f+-%.3f  PR-AUC %.3f  bal.acc(no abstain) %.3f", k,
                     v["roc_auc"]["mean"], v["roc_auc"]["std"], v["pr_auc"]["mean"], v["bal_acc_no_abstain"]["mean"])

    best = max(model_names, key=lambda k: summary["missense_only"]["models"][k]["roc_auc"]["mean"])
    log.info("selected model (best missense ROC-AUC): %s", best)
    for sub in ("missense_only", "all_in_scope"):
        c = conf[best][sub][str(ALPHA)]
        log.info("conformal alpha=%.2f [%s]: coverage %.3f abstain %.3f acc_decided %.3f sens %.3f spec %.3f",
                 ALPHA, sub, c["coverage"], c["abstain_rate"], c["acc_decided"], c["sens_decided"], c["spec_decided"])

    # ------------------------------------------------ final model + predictions on VUS
    is_cal = split_groups(lab["group"].values, CAL_FRAC, FINAL_SEED)
    proper, cal = lab[~is_cal], lab[is_cal]
    Xp, Xc = make_X(proper, proper), make_X(cal, proper)
    yp, yc = (proper.label == "R").astype(int).values, (cal.label == "R").astype(int).values
    final = make_model(best, FINAL_SEED).fit(Xp, yp)
    q = conformal_thresholds(final.predict_proba(Xc)[:, 1], yc, bin_of(cal["effect"]), ALPHA)
    Xu = make_X(unc, proper)
    pu = final.predict_proba(Xu)[:, 1]
    dec, _, _ = conformal_decide(pu, bin_of(unc["effect"]), q)
    pred = unc[["drug", "gene", "mutation", "variant", "effect"]].copy()
    pred["p_resistant"], pred["decision"] = pu.round(4), dec
    pred.to_csv(os.path.join(ROOT, "reports", "vus_predictions.csv"), index=False)
    counts = pred["decision"].value_counts().to_dict()
    log.info("uncertain variants (n=%d): %s", len(pred), counts)

    artifact = {
        "model": final, "model_name": best, "alpha": ALPHA, "thresholds": q,
        "reference": proper[["drug", "gene", "ctype", "pos", "has_pos", "group", "label"]].reset_index(drop=True),
        "n_train": int(len(proper)), "n_cal": int(len(cal)),
    }
    joblib.dump(artifact, os.path.join(ROOT, "models", "vus_guard.joblib"))
    pd.DataFrame(base.loc[:, ["drug", "gene", "mutation", "variant", "effect", "grade"]]).to_csv(
        os.path.join(ROOT, "models", "catalogue_lookup.csv"), index=False)

    out = {"summary": summary, "conformal": conf, "selected_model": best, "alpha": ALPHA,
           "final_predictions_on_uncertain": counts, "n_labelled": int(len(lab)),
           "n_uncertain": int(len(unc)), "seeds": SEEDS, "n_folds": N_FOLDS}
    with open(os.path.join(ROOT, "reports", "metrics.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    # per-row out-of-fold predictions of the selected model (seed 0) for plotting
    rec0 = records[0][best]
    oof = lab[["drug", "gene", "mutation", "effect", "label"]].copy()
    oof["p"] = rec0["p"]
    oof["decision"] = rec0["dec"][ALPHA]
    oof.to_csv(os.path.join(ROOT, "reports", "oof_predictions.csv"), index=False)
    log.info("done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
