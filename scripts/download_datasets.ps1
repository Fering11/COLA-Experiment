$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$dataRoot = Join-Path $root 'data\external'
New-Item -ItemType Directory -Force $dataRoot | Out-Null

function Save-Url($url, $path) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $path) | Out-Null
    Invoke-WebRequest -Uri $url -OutFile $path
    $hash = (Get-FileHash $path -Algorithm SHA256).Hash
    "$(Get-Date -Format o)`t$hash`t$url" | Add-Content (Join-Path $dataRoot 'SOURCES.tsv')
    Write-Host "Downloaded $path ($hash)"
}

# SEM16 training data; this is a development artifact, not the hidden test set.
Save-Url `
  'https://raw.githubusercontent.com/emsrc/SemEval2016_T6_Stance_Detection/master/semeval2016-task6-trainingdata-utf-8.txt' `
  (Join-Path $dataRoot 'sem16\semeval2016-task6-trainingdata-utf-8.txt')

# VAST project snapshot. Inspect data/ after extraction because upstream layout may change.
$vastZip = Join-Path $dataRoot 'vast\zero-shot-stance.zip'
Save-Url 'https://codeload.github.com/emilyallaway/zero-shot-stance/zip/refs/heads/master' $vastZip
$vastDir = Join-Path $dataRoot 'vast\repo'
if (Test-Path $vastDir) { Remove-Item $vastDir -Recurse -Force }
Expand-Archive $vastZip (Join-Path $dataRoot 'vast\unpacked') -Force
Write-Host "VAST snapshot extracted under $(Join-Path $dataRoot 'vast\unpacked')"

Write-Host 'P-Stance is intentionally not auto-downloaded: Twitter/X text may require hydration and license review.'
Write-Host 'See docs/dataset-sources.md for official pages and provenance rules.'
