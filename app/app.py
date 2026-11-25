"""VUS-Guard demo app.  Run:  streamlit run app/app.py"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from predict import DISCLAIMER, load, predict_one  # noqa: E402

st.set_page_config(page_title="VUS-Guard", page_icon="🧬", layout="centered")


st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #071522, #102B3A);
    color: #EAF4F4;
}

/* Main content width */
.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */
h1 {
    color: #55D6C2 !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
}

/* Section headings */
h2, h3 {
    color: #8BE3D5 !important;
}

/* Input fields */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #193747;
    border: 1px solid #397080;
    border-radius: 10px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #087F8C, #19B5A5);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #20C5B2;
    color: #071522;
    border: none;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #193747;
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #285466;
}

/* Subtle blue molecular background */
.stApp {
    background-image:
        radial-gradient(circle at 85% 20%, rgba(35, 150, 220, 0.18), transparent 25%),
        radial-gradient(circle at 90% 65%, rgba(40, 110, 210, 0.15), transparent 28%),
        linear-gradient(135deg, #071522, #102B3A);
    background-attachment: fixed;
}

/* Soft scientific glow */
.stApp::before {
    content: "";
    position: fixed;
    top: 10%;
    right: 3%;
    width: 280px;
    height: 480px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.25;
    background:
        radial-gradient(circle, #39BDEB 3px, transparent 5px) 20px 20px / 55px 65px,
        radial-gradient(circle, #248ED0 2px, transparent 5px) 45px 45px / 65px 75px;
}

</style>
""", unsafe_allow_html=True)



@st.cache_resource
def get_model():
    return load()



art, cat = get_model()

st.title("🧬 VUS-GUARD")
st.caption("Ishwari Bhawalkar | Computational Genomics & Bioinformatics")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Missense ROC-AUC", "0.962", "Model performance")

with col2:
    st.metric("Prediction Outcomes", "3", "R / S / Abstain")

with col3:
    st.metric("Reference Dataset", "WHO 2023", "Mutation catalogue")

st.markdown("---")
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
