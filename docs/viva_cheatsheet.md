# Questions you must be able to answer
1. **Why drop synonymous mutations?** 96% of S labels are synonymous and none of the uncertain ones are. Keeping them
   makes the score meaningless.
2. **What is a grouped CV and why?** All rows sharing gene+position stay in one fold, so test positions are unseen.
3. **Why not use Sens/PPV/OR columns?** They are computed from the same lab phenotypes as the labels: leakage.
4. **What are the hotspot features and how do you avoid leaking them?** Distance to the nearest known R/S position in
   the same drug-gene, computed from training rows only, with the row's own position removed.
5. **What is conformal prediction?** Calibrate on held-out data so that the true class is in the prediction set
   about 90% of the time (per class and variant type). One class in the set -> answer; two or none -> abstain.
6. **Why is leave-gene-out AUC only 0.54?** The model learns gene-specific hotspots; it cannot transfer to new genes. We say so and abstain there.
7. **Why is a baseline at 0.94 not a problem?** Honest framing: the contribution is the calibrated abstention and
   the out-of-domain guard, plus a modest accuracy gain.
8. **Can doctors use it?** No. It is a research prototype; predictions on uncertain mutations cannot be verified without lab data.
9. **What is prior shift?** Labelled data are R-rich, uncertain data probably are not, so true precision for R calls will be lower.
10. **What would you do next?** Add structure-based features (ESM embeddings / ddG), validate against CRyPTIC phenotypes, audit by lineage.
