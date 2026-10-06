param([ValidateSet("cu128","cpu")][string]$Compute = "cu128")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
Write-Host "Project: $PSScriptRoot"
& py -3.12 --version
if ($LASTEXITCODE -ne 0) { throw "Windows Python 3.12 was not found. Install Python 3.12 x64 first." }
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (!(Test-Path -LiteralPath $Python)) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip update failed." }
& $Python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url "https://download.pytorch.org/whl/$Compute"
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed." }
& $Python 00_check_environment.py
if ($LASTEXITCODE -ne 0) { throw "Device computation check failed. Read the error above." }
Write-Host "READY. In VS Code, select: $Python"
Write-Host "Packages installed only. Check BUNDLE_STATUS.json for image-data availability."
