# VUS-Guard
**Uncertainty-aware AI for classifying unclassified *M. tuberculosis* drug-resistance mutations**

> Research prototype built for the Open Health Innovation track (Track 2). **Not clinically validated. Not medical advice.**

## Problem
Sequencing-based TB drug-resistance testing keeps finding mutations the WHO catalogue grades as
*uncertain significance* (33,906 drug-mutation pairs in the 2023 catalogue). A clinician cannot act on them.
Naive ML tools answer confidently anyway. VUS-Guard predicts **resistant / susceptible / abstain** and
abstains (send for lab testing) when the evidence is not strong enough.

## Data (open access)
WHO *Catalogue of mutations in Mycobacterium tuberculosis complex and their association with drug resistance*,
2nd edition (2023), [GTB-tbsequencing/mutation-catalogue-2023](https://github.com/GTB-tbsequencing/mutation-catalogue-2023),
licence **ODC-By v1.0** (attribution required: WHO). No patient-level or personal data is used.

Labels: grades 1-2 = resistant (R), grades 4-5 = susceptible (S), grade 3 = uncertain (prediction target, no ground truth).

**Scope decision.** 96% of S labels are *synonymous* mutations and none of the uncertain mutations are synonymous.
Training on them would give a trivially high score, so they are excluded. Result: 1,800 labelled rows
(R=1,383, S=417); the hard subset is **630 missense** (R=411, S=219).
Columns computed from lab phenotypes in the WHO sheet (Sens, PPV, OR...) are **not used as features** (label leakage).

## Method
1. **Features** (`src/features.py`): drug, gene, variant class, position, BLOSUM62, hydropathy/volume/charge change,
   plus *hotspot* features: distance to the nearest resistant/susceptible labelled position in the same drug-gene
   (computed only from training data, own position left out).
2. **Models**: logistic regression and random forest; gene-prior and hotspot-only baselines.
3. **Validation**: repeated (3 x 5-fold) **grouped CV**, so a gene+position is never in both train and test.
4. **Abstention**: class-conditional split-conformal prediction (alpha = 0.10). Prediction set {R} -> R, {S} -> S, otherwise abstain.
5. **Sanity checks** (`src/sanity_checks.py`): label-shuffle test and leave-gene-out test.

## Results (missense, mean over 3 seeds; see `reports/metrics.json`)
| | ROC-AUC |
|---|---|
| Gene prior only (baseline) | 0.89 |
| Hotspot distance only (baseline) | 0.94 |
| Random forest | 0.96 |
| **Logistic regression (selected)** | **0.96** (PR-AUC 0.98) |
| Label-shuffled (should be chance) | 0.48 |
| **Unseen genes (leave-gene-out)** | **0.54** |

Conformal, alpha = 0.10, missense: coverage 91.9% (target 90%), abstains on 6.6%, balanced accuracy on answered
mutations 91.5% vs 90.3% without abstention. Figures: `reports/fig1..3`.
Model output on the 33,906 uncertain mutations: 30,431 S / 1,744 R / 1,731 abstain (`reports/vus_predictions.csv`).

## Limitations (please read)
- **Does not generalise to new genes** (AUC ~0.54). It learns hotspot patterns *within* genes that have labelled
  mutations. The app abstains when a drug-gene has no labelled reference.
- **The hotspot-only baseline is already strong (0.94).** The extra biochemical features add ~0.02 AUC; the main
  contribution is the safety layer (abstention + out-of-domain guard), not a big accuracy jump.
- **Label prior shift.** Labelled mutations are 65% resistant among missense; the uncertain set is likely far less
  resistant. Real precision for "R" calls on uncertain mutations will be lower than CV suggests. Predictions are
  **hypotheses to prioritise for lab testing**, not verified calls.
- Small data (630 missense). Labels come from the same lab data as the catalogue; no external clinical cohort was used.
- No lineage-level fairness audit (needs per-isolate data). Catalogue isolates are not representative of every region, including India.

## Run
```bash
./run_all.sh                 # ~70 s: data -> train/CV -> sanity checks -> figures
streamlit run app/app.py     # demo
```
Python 3.13, fixed seeds, training log in `logs/training_log.txt`.

## Ethics, privacy, safety
Open aggregated data only; nothing is collected or stored. Decision support only: abstention routes uncertain
cases to phenotypic testing; the clinician remains responsible for decisions.

## AI-assistance disclosure
Code was developed with Claude (Anthropic) during the hackathon week. See `AI_USAGE.md`.
