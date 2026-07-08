param(
  [string]$Destination = "third_party\MNN"
)

$ErrorActionPreference = "Stop"
$commit = "0bff03cbef43c783f44e41484b9f8a0b28bd758d"
if (-not (Test-Path $Destination)) {
  git clone --depth 1 https://github.com/alibaba/MNN.git $Destination
}
if (-not (Test-Path (Join-Path $Destination ".git"))) {
  if ((Get-ChildItem -Force $Destination | Measure-Object).Count -ne 0) {
    throw "$Destination exists but is not a git checkout and is not empty."
  }
  Remove-Item -LiteralPath $Destination -Force
  git clone --depth 1 https://github.com/alibaba/MNN.git $Destination
}
git -C $Destination fetch --depth 1 origin $commit
git -C $Destination checkout $commit
git -C $Destination rev-parse HEAD
