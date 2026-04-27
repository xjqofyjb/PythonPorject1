param(
    [string]$PythonExe = "C:\Users\researcher\miniconda3\python.exe",
    [string]$ProjectRoot = "E:\PythonProject1",
    [int]$AttachMainPid = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resultsDir = Join-Path $ProjectRoot "results"
$logsDir = Join-Path $resultsDir "logs"
$statusPath = Join-Path $resultsDir "paper_pipeline_status.json"
$traceLog = Join-Path $logsDir "paper_pipeline_trace.log"

New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

function Write-Trace {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $traceLog -Value "$stamp | $Message"
}

function Set-Status {
    param(
        [string]$Stage,
        [string]$State,
        [string]$Detail = ""
    )
    $payload = [ordered]@{
        updated_at = (Get-Date).ToString("s")
        stage = $Stage
        state = $State
        detail = $Detail
    }
    $payload | ConvertTo-Json | Set-Content -Path $statusPath -Encoding UTF8
    Write-Trace "$Stage | $State | $Detail"
}

function Invoke-Step {
    param(
        [string]$Stage,
        [string[]]$StepArgs
    )
    Set-Status -Stage $Stage -State "running" -Detail ($StepArgs -join " ")
    Write-Trace "START $Stage"
    & $PythonExe @StepArgs
    if ($LASTEXITCODE -ne 0) {
        Set-Status -Stage $Stage -State "failed" -Detail "exit_code=$LASTEXITCODE"
        throw "Stage $Stage failed with exit code $LASTEXITCODE"
    }
    Set-Status -Stage $Stage -State "completed"
    Write-Trace "DONE $Stage"
}

function Wait-ForMain {
    param([int]$TargetPid)
    if ($TargetPid -le 0) {
        return
    }
    try {
        $proc = Get-Process -Id $TargetPid -ErrorAction Stop
        Set-Status -Stage "main_wait" -State "running" -Detail "Waiting for PID $TargetPid ($($proc.ProcessName))"
        Write-Trace "Waiting for existing main PID $TargetPid"
        Wait-Process -Id $TargetPid
        Set-Status -Stage "main_wait" -State "completed" -Detail "PID $TargetPid finished"
    } catch {
        Set-Status -Stage "main_wait" -State "completed" -Detail "PID $TargetPid not found; continuing"
    }
}

Push-Location $ProjectRoot
try {
    Set-Status -Stage "bootstrap" -State "running" -Detail "Preparing background paper pipeline"
    Wait-ForMain -TargetPid $AttachMainPid

    Invoke-Step -Stage "main_table" -StepArgs @("-m", "analysis.make_tables", "--in", "results/results_main_rigorous.csv", "--out", "results/main_table.tex", "--experiment", "main")
    Invoke-Step -Stage "main_plots" -StepArgs @("-m", "analysis.make_plots", "--in", "results/results_main_rigorous.csv", "--outdir", "figs/main", "--experiment", "main")

    Invoke-Step -Stage "scenario_run" -StepArgs @("-m", "src.runner", "--config", "configs/scenario.yaml")
    Invoke-Step -Stage "scenario_table" -StepArgs @("-m", "analysis.make_tables", "--in", "results/results_scenario_rigorous.csv", "--out", "results/scenario_table.tex", "--experiment", "scenario")
    Invoke-Step -Stage "scenario_plots" -StepArgs @("-m", "analysis.make_plots", "--in", "results/results_scenario_rigorous.csv", "--outdir", "figs/scenario", "--experiment", "scenario")

    Invoke-Step -Stage "simops_run" -StepArgs @("-m", "src.runner", "--config", "configs/simops.yaml")
    Invoke-Step -Stage "simops_table" -StepArgs @("-m", "analysis.make_tables", "--in", "results/results_simops_rigorous.csv", "--out", "results/simops_table.tex", "--experiment", "simops")
    Invoke-Step -Stage "simops_plots" -StepArgs @("-m", "analysis.make_plots", "--in", "results/results_simops_rigorous.csv", "--outdir", "figs/simops", "--experiment", "simops")

    Invoke-Step -Stage "sensitivity_run" -StepArgs @("-m", "src.runner", "--config", "configs/sensitivity.yaml")
    Invoke-Step -Stage "sensitivity_table" -StepArgs @("-m", "analysis.make_tables", "--in", "results/results_sensitivity_rigorous.csv", "--out", "results/sensitivity_table.tex", "--experiment", "sensitivity")
    Invoke-Step -Stage "sensitivity_plots" -StepArgs @("-m", "analysis.make_plots", "--in", "results/results_sensitivity_rigorous.csv", "--outdir", "figs/sensitivity", "--experiment", "sensitivity")

    Invoke-Step -Stage "mechanism_run" -StepArgs @("-m", "src.runner", "--config", "configs/mechanism.yaml")
    Invoke-Step -Stage "mechanism_table" -StepArgs @("-m", "analysis.make_tables", "--in", "results/results_mechanism_rigorous.csv", "--out", "results/mechanism_table.tex", "--experiment", "mechanism")
    Invoke-Step -Stage "mechanism_plots" -StepArgs @("-m", "analysis.make_plots", "--in", "results/results_mechanism_rigorous.csv", "--outdir", "figs/mechanism", "--experiment", "mechanism")

    Invoke-Step -Stage "ablation_run" -StepArgs @("-m", "src.runner", "--config", "configs/ablation.yaml")
    Invoke-Step -Stage "ablation_plots" -StepArgs @("-m", "analysis.make_plots", "--in", "results/results_ablation_rigorous.csv", "--outdir", "figs/paper", "--experiment", "paper", "--traces_dir", "results/traces")

    Invoke-Step -Stage "paper_figures" -StepArgs @("-m", "analysis.build_paper_figures", "--results_dir", "results", "--outdir", "figs/paper")
    Set-Status -Stage "pipeline" -State "completed" -Detail "All paper data and figures generated"
} catch {
    $msg = $_.Exception.Message
    Set-Status -Stage "pipeline" -State "failed" -Detail $msg
    throw
} finally {
    Pop-Location
}
