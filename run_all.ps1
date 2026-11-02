$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
& $py -m pytest -q
Push-Location scripts
foreach ($s in "01_audit.py", "02_scam_patterns.py", "03_toxicity.py", "04_sentiment.py", "05_policy_and_transfer.py") {
    Write-Host "== $s"
    & $py $s | Out-Null
}
Pop-Location

