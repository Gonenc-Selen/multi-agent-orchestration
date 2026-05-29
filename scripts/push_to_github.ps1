# push_to_github.ps1
# Usage: .\scripts\push_to_github.ps1 ["optional commit message"]

param(
    [string]$Message = ""
)

$REMOTE_URL = "https://github.com/Gonenc-Selen/multi-agent-orchestration.git"
$BRANCH     = "main"

Set-Location "$PSScriptRoot\.."

# --- Init repo on first run ---
if (-not (Test-Path ".git")) {
    Write-Host "Initializing git repository..." -ForegroundColor Cyan
    git init
    git remote add origin $REMOTE_URL
    git checkout -b $BRANCH
    Write-Host "Git repo initialized and remote set." -ForegroundColor Green
}

# --- Stage all tracked/new files (respects .gitignore) ---
git add -A

$status = git status --porcelain
if (-not $status) {
    Write-Host "Nothing to commit, working tree clean." -ForegroundColor Yellow
    exit 0
}

# --- Build commit message ---
if ($Message -eq "") {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    $Message   = "update: $timestamp"
}

git commit -m $Message

# --- Push ---
Write-Host "Pushing to $REMOTE_URL ($BRANCH)..." -ForegroundColor Cyan
git push -u origin $BRANCH

if ($LASTEXITCODE -eq 0) {
    Write-Host "Done. Changes pushed to GitHub." -ForegroundColor Green
} else {
    Write-Host "Push failed. Check your credentials or remote URL." -ForegroundColor Red
    exit 1
}
