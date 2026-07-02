$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root "Py6SV\envs\py6s\python.exe"
$OutDir = Join-Path $Root "py6s_only\outputs"
$Surface = Join-Path $OutDir "ljn_example_surface_reflectance.csv"
$Summary = Join-Path $OutDir "ljn_example_surface_reflectance_summary.json"
$Issues = Join-Path $OutDir "ljn_example_surface_reflectance_issues.csv"
$Audit = Join-Path $OutDir "ljn_example_surface_reflectance_audit.json"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

& $Python (Join-Path $Root "py6s_only\ljn_ocid_py6s_correction.py") `
    --modis-csv (Join-Path $Root "..\Data\ljn\modis_l1b_result.csv") `
    --aeronet-csv (Join-Path $Root "..\Data\ljn\lwn_with_aod_inv15_ocid.csv") `
    --max-rows 1 `
    --bands 412 443 555 `
    --output $Surface `
    --summary-json $Summary
if ($LASTEXITCODE -ne 0) {
    throw "LJN Py6S correction failed with exit code $LASTEXITCODE"
}

& $Python (Join-Path $Root "py6s_only\tools\audit_surface_reflectance_output.py") `
    --input $Surface `
    --issues-output $Issues `
    --json-output $Audit
if ($LASTEXITCODE -ne 0) {
    throw "Surface reflectance audit failed with exit code $LASTEXITCODE"
}

Write-Host "Surface reflectance: $Surface"
Write-Host "Summary JSON: $Summary"
Write-Host "Audit JSON: $Audit"
