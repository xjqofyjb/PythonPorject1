# TR-Style Check Report

1. Old weak quick-CG data were excluded from final Table 8, Fig. 3, Fig. 5, and Fig. 6. Corrected revision-only CSVs are used for optional carbon and perturbation plots.
2. Large-scale Pool LP-IP gap labels are preserved for N=200 and N=500.
3. N=500 is marked as U-only in Fig. 3 and Table 8.
4. Method names are standardized: CG+IR, Fix-and-Optimize, Rolling-Horizon, Restricted-CG, FIFO, Greedy.
5. Captions match the generated panels and plotted data.
6. No hard-coded numerical arrays were used; plotted data are read from CSV files. Styling constants such as colors, markers, and reference policy points are fixed style choices.
7. Fig. 10 was not generated because corrected empirical-boundary data were unavailable; a missing-data report was written instead.
