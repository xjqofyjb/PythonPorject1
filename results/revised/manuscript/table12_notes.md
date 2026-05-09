# Table 12 Notes

Source CSV: `results\experiment_3_followup\aggregated.csv`.

Polishing actions:
- Kept the corrected numerical values from the existing enrichment follow-up output.
- Standardized the method name to `CG+IR`.
- Replaced informal pricing labels with manuscript terminology: `Full-pool diagnostic` and `Budgeted top-K pricing`.
- Added explicit gap-interpretation labels.
- Added the large-scale generated-pool caveat for N=200 and N=500.

Large-scale caveat:
For N=200 and N=500, the CG+IR rows are generated-pool budgeted-CG evidence rather than complete-column global optimality certificates.

Rows included:
- N=100: CG+IR, Full-pool diagnostic, Full-CG LP-IP gap
- N=200: CG+IR, Budgeted top-K pricing, Pool LP-IP gap
- N=500: CG+IR, Budgeted top-K pricing, Pool LP-IP gap
