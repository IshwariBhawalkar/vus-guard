"""VUS-Guard demo app.  Run:  streamlit run app/app.py"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from predict import DISCLAIMER, load, predict_one  # noqa: E402

st.set_page_config(page_title="VUS-Guard", page_icon="🧬", layout="centered")


@st.cache_resource
def get_model():
    return load()


art, cat = get_model()
st.title("VUS-Guard")
st.caption("Uncertainty-aware prediction for unclassified *M. tuberculosis* drug-resistance mutations")
st.warning(DISCLAIMER)

drug = st.selectbox("Drug", sorted(cat.drug.unique()))
genes = sorted(cat[cat.drug == drug].gene.unique())
gene = st.selectbox("Gene", genes)
mutation = st.text_input("Mutation (HGVS-style)", value="p.Leu430Val",
                         help="Examples: p.Ser450Leu, c.-15C>T, n.1401A>G, p.Gly12fs")

if st.button("Classify", type="primary"):
    r = predict_one(art, cat, drug, gene, mutation)
    if r["status"] == "in_catalogue":
        st.info(r["message"])
    elif r["status"] in ("error", "out_of_scope"):
        st.error(r["message"])
    else:
        p, dec = r["p_resistant"], r["decision"]
        if dec == "R":
            st.success("Prediction: likely associated with RESISTANCE")
        elif dec == "S":
            st.success("Prediction: likely NOT associated with resistance")
        else:
            st.warning("ABSTAIN: the model is not confident enough. Send the isolate for phenotypic drug-susceptibility testing.")
        st.progress(min(max(p, 0.0), 1.0), text=f"Model score for resistance: {p:.2f} (a score, not a calibrated probability)")
        if r["note"]:
            st.caption(r["note"])
        if r["was_in_catalogue_as_uncertain"]:
            st.caption("WHO catalogue currently grades this mutation as 'uncertain significance'.")
        st.caption(f"Variant class inferred: {r['effect']}. Decision rule: class-conditional conformal prediction "
                   f"(alpha = {art['alpha']}), model: {art['model_name']}.")

with st.expander("How reliable is this?"):
    st.markdown(
        "- Validated only on **new positions inside genes that have labelled mutations** (grouped cross-validation, ROC-AUC ~0.96 on missense).\n"
        "- **Does not transfer to unseen genes** (leave-gene-out AUC ~0.54), so the app abstains there.\n"
        "- Labels come from the WHO catalogue (2023); predictions on its uncertain mutations cannot be verified without lab data.\n"
        "- Data: WHO TB mutation catalogue v2, ODC-By licence. No patient data is used or stored.")
