"""Single-mutation inference used by the Streamlit app (and testable from the shell)."""
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import IN_SCOPE_EFFECTS, build_base_table  # noqa: E402
from train import ROOT, bin_of, conformal_decide, make_X  # noqa: E402

DISCLAIMER = "Research prototype. Not clinically validated. Not medical advice."


def load():
    art = joblib.load(os.path.join(ROOT, "models", "vus_guard.joblib"))
    cat = pd.read_csv(os.path.join(ROOT, "models", "catalogue_lookup.csv"))
    return art, cat


def infer_effect(mutation: str):
    """Guess the variant effect class from an HGVS-like string (the app has no annotator)."""
    m = mutation.strip()
    if m.startswith("p."):
        if "fs" in m:
            return "frameshift"
        if re.search(r"(\*|Ter)$", m):
            return "stop_gained"
        if re.search(r"del", m):
            return "inframe_deletion"
        if re.search(r"ins", m):
            return "inframe_insertion"
        if re.fullmatch(r"p\.[A-Za-z]{3}\d+=", m):
            return "synonymous_variant"
        mm = re.fullmatch(r"p\.([A-Za-z]{3})\d+([A-Za-z]{3})", m)
        if mm:
            return "synonymous_variant" if mm.group(1) == mm.group(2) else "missense_variant"
    elif m.startswith("c.-"):
        return "upstream_gene_variant"
    elif m.startswith("n."):
        return "non_coding_transcript_exon_variant"
    return None


def predict_one(art, cat, drug, gene, mutation):
    mutation = mutation.strip()
    hit = cat[(cat.drug == drug) & (cat.gene == gene) & (cat.mutation == mutation)]
    if len(hit):
        g = hit.iloc[0]["grade"]
        if not g.startswith("3"):
            return {"status": "in_catalogue", "grade": g,
                    "message": f"Already classified by WHO: {g}. No prediction needed."}
    effect = infer_effect(mutation)
    if effect is None:
        return {"status": "error", "message": "Could not parse the mutation. Use forms like p.Ser450Leu, c.-15C>T, n.1401A>G, p.Gly12fs."}
    if effect not in IN_SCOPE_EFFECTS:
        return {"status": "out_of_scope", "message": "Synonymous changes are out of scope (the catalogue treats them as not associated with resistance)."}
    row = pd.DataFrame([{"drug": drug, "gene": gene, "mutation": mutation, "variant": f"{gene}_{mutation}",
                         "effect": effect, "lab": "U", "final_grade": "3) Uncertain significance"}])
    base = build_base_table(row)
    ref = art["reference"]
    known_region = ((ref.drug == drug) & (ref.gene == gene) & (ref.ctype == base.loc[0, "ctype"])).any()
    p = float(art["model"].predict_proba(make_X(base, ref))[:, 1][0])
    dec, _, _ = conformal_decide(np.array([p]), bin_of([effect]), art["thresholds"])
    decision = str(dec[0])
    note = ""
    if not known_region:
        decision = "abstain"
        note = "No labelled reference mutations for this drug/gene. The model does not transfer to unseen genes (leave-gene-out AUC ~0.54), so it abstains."
    return {"status": "predicted", "effect": effect, "p_resistant": p, "decision": decision, "note": note,
            "was_in_catalogue_as_uncertain": bool(len(hit))}
