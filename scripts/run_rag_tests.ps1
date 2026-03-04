param(
    [ValidateSet("rag", "unit", "all")]
    [string]$Mode = "rag",
    [switch]$RebuildIndex,
    [switch]$AllUnitTests,
    [switch]$SkipHealthCheck
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Python not found at '$pythonExe'. Please create/activate .venv first."
}

$env:PYTHONIOENCODING = "utf-8"

function Get-EnvValue {
    param(
        [string]$Key,
        [string]$DefaultValue
    )

    if (-not (Test-Path ".env")) {
        return $DefaultValue
    }

    $line = Get-Content ".env" | Where-Object { $_ -match "^\s*$Key\s*=" } | Select-Object -First 1
    if (-not $line) {
        return $DefaultValue
    }

    return ($line -split "=", 2)[1].Trim()
}

function Invoke-Python {
    param(
        [string[]]$PythonArgs,
        [string]$PipeInputText
    )

    if ([string]::IsNullOrEmpty($PipeInputText)) {
        & $pythonExe @PythonArgs
    } else {
        $PipeInputText | & $pythonExe @PythonArgs
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $pythonExe $($PythonArgs -join ' ')"
    }
}

function Test-OllamaHealth {
    param(
        [string]$BaseUrl,
        [string]$Model
    )

    $os = Get-CimInstance Win32_OperatingSystem
    $totalRamGb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    $freeRamGb = [math]::Round($os.FreePhysicalMemory / 1MB, 2)

    Write-Host ("RAM: total={0} GB, free={1} GB" -f $totalRamGb, $freeRamGb)
    Write-Host ("Ollama: {0} (model={1})" -f $BaseUrl, $Model)

    $tags = Invoke-RestMethod -Uri "$BaseUrl/api/tags" -Method Get -TimeoutSec 20
    $modelInfo = $tags.models | Where-Object { $_.name -eq $Model } | Select-Object -First 1

    if (-not $modelInfo) {
        throw "Model '$Model' not found in Ollama. Please pull it first."
    }

    $modelSizeGb = [math]::Round(($modelInfo.size / 1GB), 2)
    Write-Host ("Model size: ~{0} GB" -f $modelSizeGb)

    if ($freeRamGb -lt $modelSizeGb) {
        Write-Warning "Free RAM is lower than model size. Tests may fail due to memory pressure."
    }

    $body = @{
        model   = $Model
        prompt  = "health check"
        stream  = $false
        options = @{
            num_predict = 8
        }
    } | ConvertTo-Json -Depth 5

    $response = Invoke-RestMethod -Uri "$BaseUrl/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 180
    if (-not $response.response) {
        throw "Ollama health check failed: empty response."
    }

    Write-Host "Ollama health check passed."
}

$ollamaBaseUrl = Get-EnvValue -Key "OLLAMA_BASE_URL" -DefaultValue "http://localhost:11434"
$ollamaModel = Get-EnvValue -Key "OLLAMA_MODEL" -DefaultValue "llama3:8b"

if (-not $SkipHealthCheck) {
    Test-OllamaHealth -BaseUrl $ollamaBaseUrl -Model $ollamaModel
}

if ($Mode -in @("rag", "all")) {
    if ($RebuildIndex) {
        Write-Host "`nRebuilding index..."
        Invoke-Python -PythonArgs @("-m", "src.ingest")
    }

    Write-Host "`nRunning RAG integration tests..."
    Invoke-Python -PythonArgs @("test_cimb_loans.py") -PipeInputText "quit"
}

if ($Mode -in @("unit", "all")) {
    Write-Host "`nRunning unit tests..."
    if ($AllUnitTests) {
        Invoke-Python -PythonArgs @("-m", "pytest", "tests", "-q")
    } else {
        # Default to RAG-focused tests for faster feedback and fewer unrelated dependency failures.
        Invoke-Python -PythonArgs @(
            "-m", "pytest",
            "tests/test_query.py",
            "tests/test_metadata_formatting.py",
            "-q"
        )
    }
}

Write-Host "`nAll requested tests completed."
