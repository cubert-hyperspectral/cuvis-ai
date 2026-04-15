$inputDir = "D:\data\XMR_25mm_CubertParkingLotTracking"
$outputDir = "D:\experiments\20260318\video_creation\cuvis-plugin\invisible-ink"
$pluginXmlPath = "C:\Users\nima.ghorbani\CuvisNEXT\invisible_ink.xml"
$processingMode = "SpectralRadiance"
$minFileSizeGB = 1
$savePipelineConfig = $false

$minFileSizeBytes = [int64]($minFileSizeGB * 1GB)
$normalizedInputDir = [System.IO.Path]::GetFullPath($inputDir).TrimEnd('\', '/')
$runBasename = "."

if (-not (Test-Path -LiteralPath $inputDir -PathType Container)) {
    throw "Input directory not found: $inputDir"
}
if (-not (Test-Path -LiteralPath $pluginXmlPath -PathType Leaf)) {
    throw "Plugin XML not found: $pluginXmlPath"
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Get-ChildItem -Path $inputDir -Filter "Auto_*.cu3s" -File -Recurse | ForEach-Object {
    if ($_.Length -le $minFileSizeBytes) {
        Write-Host "Skipping: $($_.Name) (size $([math]::Round($_.Length / 1GB, 2)) GB <= threshold $minFileSizeGB GB)"
        return
    }

    $stem = $_.BaseName
    $inputFile = $_.FullName
    $sourceParentFullPath = [System.IO.Path]::GetFullPath($_.DirectoryName).TrimEnd('\', '/')
    if ($sourceParentFullPath.StartsWith($normalizedInputDir, [System.StringComparison]::OrdinalIgnoreCase)) {
        $relativeParent = $sourceParentFullPath.Substring($normalizedInputDir.Length).TrimStart('\', '/')
    } else {
        $relativeParent = "."
    }

    $targetOutputParent = if ($relativeParent -eq ".") {
        $outputDir
    } else {
        Join-Path $outputDir $relativeParent
    }
    $outputFile = Join-Path $targetOutputParent "$stem.mp4"
    New-Item -ItemType Directory -Force -Path $targetOutputParent | Out-Null

    Write-Host "Processing: $($_.Name) (size $([math]::Round($_.Length / 1GB, 2)) GB) -> $outputFile"

    if ($savePipelineConfig) {
        uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
            --cu3s-path $inputFile `
            --output-dir $targetOutputParent `
            --out-basename $runBasename `
            --method cuvis-plugin `
            --plugin-xml-path $pluginXmlPath `
            --processing-mode $processingMode `
            --save-pipeline-config
    } else {
        uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
            --cu3s-path $inputFile `
            --output-dir $targetOutputParent `
            --out-basename $runBasename `
            --method cuvis-plugin `
            --plugin-xml-path $pluginXmlPath `
            --processing-mode $processingMode `
            --no-save-pipeline-config
    }
}
