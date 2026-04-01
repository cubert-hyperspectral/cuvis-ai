param(
    [Parameter(Mandatory = $true)]
    [string]$InputDir,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [ValidateSet("cie_tristimulus", "cir")]
    [string]$Method = "cie_tristimulus",

    [ValidateSet("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")]
    [string]$ProcessingMode = "SpectralRadiance",

    [ValidateSet("sampled_fixed", "running", "per_frame", "live_running_fixed")]
    [string]$NormalizationMode = "sampled_fixed",

    [double]$SampleFraction = 0.05,
    [double]$MinFileSizeGB = 1,
    [switch]$SavePipelineConfig,
    [double]$NirNm = 860,
    [double]$RedNm = 670,
    [double]$GreenNm = 560
)

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$normalizedInputDir = [System.IO.Path]::GetFullPath($InputDir).TrimEnd('\', '/')
$normalizedOutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$minFileSizeBytes = [int64]($MinFileSizeGB * 1GB)

if (-not (Test-Path -LiteralPath $normalizedInputDir -PathType Container)) {
    throw "Input directory does not exist or is not a directory: $normalizedInputDir"
}

New-Item -ItemType Directory -Force -Path $normalizedOutputDir | Out-Null

$baseExporterArgs = @(
    "run",
    "python",
    "examples/object_tracking/export_cu3s_false_rgb_video.py",
    "--method",
    $Method,
    "--processing-mode",
    $ProcessingMode,
    "--normalization-mode",
    $NormalizationMode,
    "--sample-fraction",
    $SampleFraction
)

if ($Method -eq "cir") {
    $baseExporterArgs += @(
        "--nir-nm",
        $NirNm,
        "--red-nm",
        $RedNm,
        "--green-nm",
        $GreenNm
    )
}

if ($SavePipelineConfig.IsPresent) {
    $baseExporterArgs += "--save-pipeline-config"
} else {
    $baseExporterArgs += "--no-save-pipeline-config"
}

Push-Location $repoRoot
try {
    Get-ChildItem -LiteralPath $normalizedInputDir -Filter "Auto_*.cu3s" -File -Recurse |
        Sort-Object FullName |
        ForEach-Object {
            if ($_.Length -le $minFileSizeBytes) {
                Write-Host "Skipping: $($_.Name) (size $([math]::Round($_.Length / 1GB, 2)) GB <= threshold $MinFileSizeGB GB)"
                return
            }

            $stem = $_.BaseName
            $inputFile = $_.FullName
            $sourceParentFullPath = [System.IO.Path]::GetFullPath($_.DirectoryName).TrimEnd('\', '/')

            if ($sourceParentFullPath.StartsWith($normalizedInputDir, [System.StringComparison]::OrdinalIgnoreCase)) {
                $relativeParent = $sourceParentFullPath.Substring($normalizedInputDir.Length).TrimStart('\', '/')
            } else {
                $relativeParent = ""
            }

            $targetOutputParent = if ([string]::IsNullOrEmpty($relativeParent)) {
                $normalizedOutputDir
            } else {
                Join-Path $normalizedOutputDir $relativeParent
            }

            $runBasename = "."
            $outputFile = Join-Path $targetOutputParent "$stem.mp4"
            New-Item -ItemType Directory -Force -Path $targetOutputParent | Out-Null

            Write-Host "Processing: $($_.Name) (size $([math]::Round($_.Length / 1GB, 2)) GB) -> $outputFile"

            $exporterArgs = $baseExporterArgs + @(
                "--cu3s-path",
                $inputFile,
                "--output-dir",
                $targetOutputParent,
                "--out-basename",
                $runBasename
            )

            & uv @exporterArgs
        }
}
finally {
    Pop-Location
}
