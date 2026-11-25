"""Feature engineering for VUS-Guard.

Only SEQUENCE-DERIVED features are used. The WHO sheet also contains columns computed
from lab phenotypes (Sens, PPV, OR, ...). Those are deliberately NOT used: they are
derived from the very labels we try to predict and would leak the answer.
"""
import re
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

BLOSUM62 = substitution_matrices.load("BLOSUM62")

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
# Kyte-Doolittle hydropathy
HYDRO = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
             L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
# Side-chain volume in cubic angstrom (Zamyatnin)
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2,
           I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8,
           Y=193.6, V=140.0)
CHARGE = dict(K=1, R=1, H=0.1, D=-1, E=-1)

# Effects that appear among the uncertain (grade 3) variants. Synonymous variants are
# excluded: 96% of "susceptible" labels are synonymous and none are uncertain, so
# including them would make the task trivially easy and the metrics meaningless.
IN_SCOPE_EFFECTS = {
    "missense_variant", "upstream_gene_variant", "non_coding_transcript_exon_variant",
    "frameshift", "stop_gained", "inframe_deletion", "inframe_insertion", "LoF",
    "stop_lost", "feature_ablation", "start_lost",
}

CAT_COLS = ["drug", "gene", "effect", "ctype"]
NUM_COLS = ["pos", "has_pos", "blosum", "d_hydro", "abs_d_hydro", "d_vol", "abs_d_vol",
            "d_charge", "alt_is_pro", "ref_is_gly", "is_transition",
            "dR", "dS", "nR10", "nS10"]
CAP = 100.0  # distance cap (codons or nucleotides)


def parse_mutation(mutation: str) -> dict:
    """Turn an HGVS-like string (p.Ser450Leu, c.-15C>T, n.1401A>G, LoF...) into fields."""
    out = dict(ctype="other", pos=np.nan, ref_aa=None, alt_aa=None, ref_nt=None, alt_nt=None)
    m = str(mutation)
    if m.startswith("p."):
        out["ctype"] = "p"
        mm = re.match(r"p\.([A-Za-z]{3})(\d+)", m)
        if mm:
            out["pos"] = float(mm.group(2))
            out["ref_aa"] = THREE_TO_ONE.get(mm.group(1))
        alt = re.match(r"p\.[A-Za-z]{3}\d+([A-Za-z]{3})$", m)
        if alt:
            out["alt_aa"] = THREE_TO_ONE.get(alt.group(1))
    elif m.startswith(("c.", "n.")):
        out["ctype"] = m[0]
        mm = re.match(r"[cn]\.(-?\d+)", m)
        if mm:
            out["pos"] = float(mm.group(1))
        nt = re.search(r"([ACGT])>([ACGT])", m)
        if nt:
            out["ref_nt"], out["alt_nt"] = nt.group(1), nt.group(2)
    return out


def _intrinsic_features(df: pd.DataFrame) -> pd.DataFrame:
    parsed = pd.DataFrame([parse_mutation(m) for m in df["mutation"]], index=df.index)
    f = pd.DataFrame(index=df.index)
    f["ctype"] = parsed["ctype"]
    f["pos"] = parsed["pos"]
    f["has_pos"] = parsed["pos"].notna().astype(float)
    f["pos"] = f["pos"].fillna(0.0)

    def sub(row, table):
        r, a = row
        return table[a] - table[r] if (r in table and a in table) else 0.0

    pairs = list(zip(parsed["ref_aa"], parsed["alt_aa"]))
    f["blosum"] = [BLOSUM62[r][a] if (r in HYDRO and a in HYDRO) else 0.0 for r, a in pairs]
    f["d_hydro"] = [sub(p, HYDRO) for p in pairs]
    f["d_vol"] = [sub(p, VOL) for p in pairs]
    f["d_charge"] = [CHARGE.get(a, 0) - CHARGE.get(r, 0) if (r in HYDRO and a in HYDRO) else 0.0
                     for r, a in pairs]
    f["abs_d_hydro"] = f["d_hydro"].abs()
    f["abs_d_vol"] = f["d_vol"].abs()
    f["alt_is_pro"] = (parsed["alt_aa"] == "P").astype(float)
    f["ref_is_gly"] = (parsed["ref_aa"] == "G").astype(float)
    ts = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    f["is_transition"] = [1.0 if (r, a) in ts else 0.0
                          for r, a in zip(parsed["ref_nt"], parsed["alt_nt"])]
    return f


def group_key(df: pd.DataFrame) -> pd.Series:
    """Rows from the same gene+position (any drug) always stay in the same CV fold, so the
    model is tested on codons/positions it has never seen. Rows without a position
    (LoF, deletion...) are grouped by their variant name."""
    parsed = [parse_mutation(m) for m in df["mutation"]]
    keys = []
    for (_, row), p in zip(df.iterrows(), parsed):
        if np.isnan(p["pos"]):
            keys.append(f"{row['gene']}|{row['mutation']}")
        else:
            keys.append(f"{row['gene']}|{p['ctype']}|{int(p['pos'])}")
    return pd.Series(keys, index=df.index)


def neighbour_features(target: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Hotspot features, computed from a REFERENCE set of labelled rows only.

    dR / dS : distance to the nearest resistant / susceptible reference position
              (same drug, gene, coordinate type), capped at CAP.
    nR10/nS10: number of resistant / susceptible reference positions within +/-10.

    The target row's own group is excluded from the reference. Without that, a training
    row would always see itself at distance 0 and the model would learn a leaky shortcut.
    ref needs columns: drug, gene, ctype, pos, has_pos, group, label ('R'/'S').
    """
    out = np.zeros((len(target), 4))
    out[:, 0] = CAP
    out[:, 1] = CAP
    ref = ref[ref["has_pos"] > 0]
    lookup = {k: (g["pos"].values, g["label"].values, g["group"].values)
              for k, g in ref.groupby(["drug", "gene", "ctype"])}
    tvals = target[["drug", "gene", "ctype", "pos", "has_pos", "group"]].values
    for i, (drug, gene, ctype, pos, has_pos, grp) in enumerate(tvals):
        if has_pos <= 0 or (drug, gene, ctype) not in lookup:
            continue
        rpos, rlab, rgrp = lookup[(drug, gene, ctype)]
        keep = rgrp != grp                      # leave the row's own group out
        rpos, rlab = rpos[keep], rlab[keep]
        for j, lab in ((0, "R"), (1, "S")):
            p = rpos[rlab == lab]
            if len(p):
                d = np.abs(p - pos)
                out[i, j] = min(d.min(), CAP)
                out[i, 2 + j] = int((d <= 10).sum())
    return pd.DataFrame(out, index=target.index, columns=["dR", "dS", "nR10", "nS10"])


def build_base_table(df: pd.DataFrame) -> pd.DataFrame:
    """Intrinsic (reference-free) features + identifiers + label."""
    base = _intrinsic_features(df)
    base["drug"] = df["drug"].values
    base["gene"] = df["gene"].values
    base["effect"] = df["effect"].values
    base["mutation"] = df["mutation"].values
    base["variant"] = df["variant"].values
    base["group"] = group_key(df).values
    base["label"] = df["lab"].values
    base["grade"] = df["final_grade"].values
    return base
