$python = "C:\Users\researcher\miniconda3\python.exe"
$workdir = "E:\PythonProject1"
$log = "E:\PythonProject1\results\logs\remaining_nonmain_baselines_20260331.log"

Set-Location $workdir
"[$(Get-Date -Format s)] Start remaining non-main baseline runs" | Out-File -FilePath $log -Encoding utf8

& $python -m src.runner --config configs/sensitivity_new_baselines_battery_08.yaml *>> $log
& $python -m src.runner --config configs/sensitivity_new_baselines_shore.yaml *>> $log
& $python -m src.runner --config configs/sensitivity_new_baselines_deadline.yaml *>> $log
& $python -m src.runner --config configs/simops_refresh_with_baselines.yaml *>> $log

& $python -m analysis.make_tables --in results/results_scenario_rigorous.csv --out results/scenario_table.tex --experiment scenario *>> $log
& $python -m analysis.make_tables --in results/results_simops_rigorous.csv --out results/simops_table.tex --experiment simops *>> $log
& $python -m analysis.make_tables --in results/results_sensitivity_rigorous.csv --out results/sensitivity_table.tex --experiment main *>> $log
& $python -m analysis.make_tables --in results/results_mechanism_rigorous.csv --out results/mechanism_table.tex --experiment main *>> $log
& $python -m analysis.build_paper_figures --results_dir results --outdir figs/paper *>> $log

"[$(Get-Date -Format s)] Finished remaining non-main baseline runs" | Out-File -FilePath $log -Encoding utf8 -Append
