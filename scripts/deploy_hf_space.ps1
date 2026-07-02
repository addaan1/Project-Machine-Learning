param(
    [Parameter(Mandatory = $true)]
    [string]$SpaceId
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StageRoot = Join-Path "C:\tmp" "ecodash-hf-space"

$HfCommand = Get-Command hf -ErrorAction SilentlyContinue
if (-not $HfCommand) {
    $Candidate = Join-Path $env:APPDATA "Python\Python312\Scripts\hf.exe"
    if (Test-Path $Candidate) {
        $HfCommand = Get-Item $Candidate
    }
}

if (-not $HfCommand) {
    throw "Hugging Face CLI tidak ditemukan. Install dengan: python -m pip install --user -U huggingface_hub"
}

$HfPath = if ($HfCommand.Source) { $HfCommand.Source } else { $HfCommand.FullName }
if ($SpaceId -match "<|>") {
    throw "SpaceId masih memakai placeholder. Contoh yang benar: Adan11/ecodash-indonesia"
}

Write-Host "Checking Hugging Face authentication..."
try {
    & $HfPath auth whoami | Out-Host
} catch {
    throw "Belum login ke Hugging Face. Jalankan: `"$HfPath`" auth login"
}

Write-Host "Creating or reusing Space $SpaceId..."
& $HfPath repos create $SpaceId --type space --space-sdk docker --flavor cpu-basic --public --exist-ok

Write-Host "Configuring production environment variables..."
& $HfPath spaces variables add $SpaceId `
    -e "DJANGO_ENV=production" `
    -e "DEBUG=False" `
    -e "ALLOWED_HOSTS=.hf.space,localhost,127.0.0.1,0.0.0.0" `
    -e "CSRF_TRUSTED_ORIGINS=https://*.hf.space"

$GeneratedSecret = -join ((1..4) | ForEach-Object { [guid]::NewGuid().ToString("N") })
& $HfPath spaces secrets add $SpaceId -s "SECRET_KEY=$GeneratedSecret"

Write-Host "Preparing clean deploy staging folder..."
if (Test-Path $StageRoot) {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $StageRoot | Out-Null

$RuntimeDirs = @("dashboard", "datasets", "models")
foreach ($Dir in $RuntimeDirs) {
    Copy-Item -LiteralPath (Join-Path $RepoRoot $Dir) -Destination (Join-Path $StageRoot $Dir) -Recurse -Force
}

Remove-Item -LiteralPath (Join-Path $StageRoot "dashboard\staticfiles") -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $StageRoot "dashboard\db.sqlite3") -Force -ErrorAction SilentlyContinue

Copy-Item -LiteralPath (Join-Path $RepoRoot "Dockerfile") -Destination $StageRoot -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "requirements-deploy.txt") -Destination $StageRoot -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot ".dockerignore") -Destination $StageRoot -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "hf-space\README.md") -Destination (Join-Path $StageRoot "README.md") -Force

Write-Host "Uploading EcoDash deploy bundle..."
& $HfPath upload $SpaceId $StageRoot . --type space --commit-message "Deploy EcoDash Django app"

Write-Host "Deploy request sent."
Write-Host "Open: https://huggingface.co/spaces/$SpaceId"
