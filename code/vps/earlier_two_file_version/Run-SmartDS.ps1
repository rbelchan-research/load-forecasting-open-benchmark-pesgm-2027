<#
  Run-SmartDS.ps1
  Builds the transformer forecasting dataset from SMART-DS on a Windows VPS:
  downloads the feeder models, runs an OpenDSS time-series simulation, and writes
  CSV files plus one zip to send back.

  Put this file and smartds_build.py in the same folder, then:

      powershell -ExecutionPolicy Bypass -File .\Run-SmartDS.ps1 -DryRun
      powershell -ExecutionPolicy Bypass -File .\Run-SmartDS.ps1

  The dry run downloads only the model files and prints what the full run would
  pull. It takes a couple of minutes. The full run downloads several GB of customer
  profiles and then simulates, so give it a few hours and a screen session you can
  leave alone. Everything is resumable: rerun the same command and it skips files
  it already has.

  Useful switches:
      -Out C:\sds              where everything lands (keep the path short)
      -Substations 2           substations per sub-region
      -Step 60                 QSTS step in minutes; 60 is about 4x faster than 15
      -NoSimulate              skip the power flow, summation dataset only
      -Region AUS              AUS, SFO or SAF instead of Greensboro
#>

[CmdletBinding()]
param(
    [string]$Out = "C:\sds",
    [string]$Region = "GSO",
    [string]$Subs = "urban-suburban,rural",
    [int]$Substations = 2,
    [int]$Step = 15,
    [int]$Year = 2018,
    [switch]$NoSimulate,
    [switch]$DryRun,
    [switch]$InstallPython
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$worker = Join-Path $here "smartds_build.py"

function Say($m) { Write-Host "[smartds] $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[smartds] $m" -ForegroundColor Yellow }

if (-not (Test-Path $worker)) {
    throw "smartds_build.py is not next to this script. Put both files in the same folder."
}

# ---------------------------------------------------------------- find python
$py = $null
foreach ($c in @("py -3", "python", "python3")) {
    try {
        $parts = $c.Split(" ")
        $v = & $parts[0] $parts[1..($parts.Length - 1)] --version 2>&1
        if ($LASTEXITCODE -eq 0 -and "$v" -match "Python 3\.(\d+)") {
            if ([int]$Matches[1] -ge 9) { $py = $c; break }
        }
    } catch { }
}

if (-not $py) {
    if ($InstallPython) {
        Say "installing Python with winget"
        winget install --id Python.Python.3.12 --silent --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
        $py = "python"
    } else {
        Warn "No Python 3.9 or newer found."
        Warn "Install it with:  winget install --id Python.Python.3.12"
        Warn "or rerun this script with -InstallPython"
        exit 1
    }
}
Say "using python: $py"

# ---------------------------------------------------------------- environment
$venv = Join-Path $Out ".venv"
$vpy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $Out)) { New-Item -ItemType Directory -Path $Out | Out-Null }

if (-not (Test-Path $vpy)) {
    Say "creating virtual environment in $venv"
    $parts = $py.Split(" ")
    & $parts[0] $parts[1..($parts.Length - 1)] -m venv $venv
}

Say "installing packages (first run only, a minute or two)"
& $vpy -m pip install --quiet --upgrade pip
& $vpy -m pip install --quiet pandas pyarrow "opendssdirect.py"
if ($LASTEXITCODE -ne 0) { throw "package install failed" }

& $vpy -c "import opendssdirect as d; print('OpenDSS', d.Basic.Version())"

# ---------------------------------------------------------------- run
$argsList = @(
    $worker,
    "--out", $Out,
    "--region", $Region,
    "--subs", $Subs,
    "--substations", $Substations,
    "--step", $Step,
    "--year", $Year
)
if ($NoSimulate) { $argsList += "--no-simulate" }
if ($DryRun) { $argsList += "--dry-run" }

$log = Join-Path $Out ("run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Say "starting. console output is also written to $log"
Say "command: $vpy $($argsList -join ' ')"

& $vpy @argsList 2>&1 | Tee-Object -FilePath $log

if ($LASTEXITCODE -ne 0) {
    Warn "the run stopped early. Send me $log and I will fix it."
    exit $LASTEXITCODE
}

$zip = Join-Path $Out "smartds_dataset.zip"
if (Test-Path $zip) {
    $mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
    Say "done. Send me this file: $zip  ($mb MB)"
} else {
    Warn "no zip produced. Send me $log"
}
