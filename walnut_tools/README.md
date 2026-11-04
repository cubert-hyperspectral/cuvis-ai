# walnut_tools

Helper scripts that built, checked and benchmarked the walnut presets in
`cuvis_ai/configs/pipeline/walnut_*`. They live outside the `cuvis_ai` package so the wheel does not
ship them. Each folder matches its preset folder, and every script still reads and writes that
preset folder (`HERE` / `SEG` point into `cuvis_ai/configs/pipeline/<folder>`), so run them from
anywhere in the checkout, for example:

    uv run python walnut_tools/walnut_seg/check_seg_pipelines.py

Most of them need a GPU, the walnut recordings and the plugin stack; see the README in the matching
preset folder for the arguments.
