[CmdletBinding()]
param(
    [ValidateRange(40, 500)]
    [int]$OptunaTrials = 80,
    [switch]$SkipShapFairness
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$notebookDir = Join-Path $projectRoot 'notebooks'
$notebookPath = Join-Path $notebookDir 'home_credit_scoring.ipynb'
$reportDir = Join-Path $projectRoot 'reports'
$submissionPath = Join-Path $projectRoot 'submissions/submission_oof_blend.csv'
$resultsPath = Join-Path $reportDir 'final_results.json'
$pdfPath = Join-Path $reportDir 'home_credit_final_report.pdf'

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'data/raw/application_train.csv'))) {
    throw "Не найден data/raw/application_train.csv: $projectRoot"
}

# Параметры считываются Setup-ячейкой. Временные признаки и модели хранятся только в RAM.
$env:HOME_CREDIT_RUN_FULL = '1'
$env:HOME_CREDIT_OPTUNA_TRIALS = "$OptunaTrials"
$env:HOME_CREDIT_SHAP_FAIRNESS = if ($SkipShapFairness) { '0' } else { '1' }
$env:HOME_CREDIT_FAST_MODE = '0'

Push-Location $notebookDir
try {
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA не найдена'; print('GPU:', torch.cuda.get_device_name(0))"
    python -m jupyter nbconvert --to notebook --execute --inplace home_credit_scoring.ipynb --ExecutePreprocessor.timeout=-1
    python -m jupyter nbconvert --to html --output-dir $reportDir home_credit_scoring.ipynb
    if (-not $SkipShapFairness) {
        python (Join-Path $PSScriptRoot 'build_final_report.py') --results $resultsPath --output $pdfPath --figures-dir (Join-Path $reportDir 'figures')
    }
}
finally {
    Pop-Location
}

Write-Host "Готово: выполненный notebook и HTML-версия отчёта находятся в $reportDir"
if (-not $SkipShapFairness) { Write-Host "PDF к сдаче: $pdfPath" }
Write-Host "Финальный Kaggle CSV: $submissionPath"
