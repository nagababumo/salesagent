$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    $python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        throw "Virtual environment not found at .venv. Create it and install the dependencies described in README.md."
    }

    $common = @("--data-dir", "data", "--evaluate", "--ground-truth", "data\gt.txt")

    & $python "modelcodesday1\whisperltv3.py" @common "--evaluation-output" "modelbenchmark_day1\whisperevaluation_results.json"
    if ($LASTEXITCODE -ne 0) { throw "whisperltv3.py failed with exit code $LASTEXITCODE" }
    & $python "modelcodesday1\whisper_fintuned.py" @common "--evaluation-output" "modelbenchmark_day1\ftwhisper_evaluation.json"
    if ($LASTEXITCODE -ne 0) { throw "whisper_fintuned.py failed with exit code $LASTEXITCODE" }
    & $python "modelcodesday1\indicconfor.py" @common "--batch-size" "1" "--evaluation-output" "modelbenchmark_day1\indicconfor_evaluation.json"
    if ($LASTEXITCODE -ne 0) { throw "indicconfor.py failed with exit code $LASTEXITCODE" }
    & $python "modelcodesday1\sraavani.py" @common "--batch-size" "4" "--chunk-seconds" "30" "--overlap-seconds" "1" "--evaluation-output" "modelbenchmark_day1\sravaani_evaluation.json"
    if ($LASTEXITCODE -ne 0) { throw "sraavani.py failed with exit code $LASTEXITCODE" }
    & $python "modelcodesday1\transcriber_nemotron.py" @common "--lookahead-tokens" "6" "--evaluation-output" "modelbenchmark_day1\evaluation_results_nemotron.json"
    if ($LASTEXITCODE -ne 0) { throw "transcriber_nemotron.py failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}