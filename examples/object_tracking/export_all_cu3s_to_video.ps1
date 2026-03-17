$inputDir  = "D:\data\XMR_notarget_Busstation\20260226"
$outputDir = "D:\data\XMR_notarget_Busstation\20260226\videos"
$normalizationMode = "sampled_fixed"
$sampleFraction = 0.05
$minFileSizeGB = 1
$savePipelineConfig = $false
$minFileSizeBytes = [int64]($minFileSizeGB * 1GB)

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Get-ChildItem -Path $inputDir -Filter "Auto_*.cu3s" | ForEach-Object {
    if ($_.Length -le $minFileSizeBytes) {
        Write-Host "Skipping: $($_.Name) (size $([math]::Round($_.Length / 1GB, 2)) GB <= threshold $minFileSizeGB GB)"
        return
    }

    $stem       = $_.BaseName
    $inputFile  = $_.FullName
    $outputFile = Join-Path $outputDir "$stem.mp4"

    Write-Host "Processing: $($_.Name) (size $([math]::Round($_.Length / 1GB, 2)) GB) -> $outputFile"

    if ($savePipelineConfig) {
        uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
            --cu3s-file-path    $inputFile  `
            --output-video-path $outputFile `
            --method cie_tristimulus `
            --processing-mode SpectralRadiance `
            --normalization-mode $normalizationMode `
            --sample-fraction $sampleFraction `
            --save-pipeline-config
    } else {
        uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
            --cu3s-file-path    $inputFile  `
            --output-video-path $outputFile `
            --method cie_tristimulus `
            --processing-mode SpectralRadiance `
            --normalization-mode $normalizationMode `
            --sample-fraction $sampleFraction `
            --no-save-pipeline-config
    }
}
