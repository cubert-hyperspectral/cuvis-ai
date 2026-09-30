"""Build walnut_diag_passthrough_cuvisnext_cube: the control pipeline for the live-view jitter question.

cu3s cube -> FixedWavelengthSelector (CIR 870 / 640 / 550 nm, per-frame normalisation) -> `CIR.rgb_image`. No model,
a few ms per frame: if the view still steps back and forth with this pipeline running, the cause is the live-inference
path of the app, not the detectors. The `plugins:` list names patchcore and steervit (unused) so cuvis.next reuses the
FO pipelines' child env instead of composing a new one.

    <stack>/cuvis-ai/.venv python build_passthrough.py
"""

from pathlib import Path

import yaml
from cuvis_ai.node.channel_selector import FixedWavelengthSelector
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

HERE = Path(__file__).resolve().parent
NAME = "walnut_diag_passthrough_cuvisnext_cube"

pipe = CuvisPipeline(NAME)
src = CU3SDataNode(name="DataSource")
cir = FixedWavelengthSelector(target_wavelengths=[870.0, 640.0, 550.0], normalize_output=True, norm_mode="per_frame",
                              apply_gamma=False, name="CIR")
pipe.connect(src.outputs.cube, cir.inputs.cube)
pipe.connect(src.outputs.wavelengths, cir.inputs.wavelengths)
out = HERE / f"{NAME}.yaml"
pipe.save_to_file(str(out))

cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
cfg["metadata"]["name"] = NAME
cfg["metadata"]["description"] = (
    "Control pipeline for the live-view jitter test: the cube to a CIR false-colour image (870/640/550 nm, per-frame "
    "normalisation), no model. Display CIR.rgb_image. If the view steps back and forth with this running too, the "
    "cause is the app's live-inference path, not the detectors. patchcore / steervit are listed only so cuvis.next "
    "reuses the FO pipelines' child env.")
cfg["metadata"]["tags"] = ["walnut", "diagnostic", "passthrough"]
cfg["plugins"] = ["cuvis_ai_builtin", "patchcore", "steervit"]
for n in cfg["nodes"]:
    if n["class_name"].endswith("CU3SDataNode"):
        n["hparams"] = {}  # cuvis.next injects the cube
order = {k: i for i, k in enumerate(("metadata", "plugins", "nodes", "connections"))}
cfg = dict(sorted(cfg.items(), key=lambda kv: order.get(kv[0], 9)))
out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

PipelineBuilder().build_from_config(str(out))  # round trip
print("wrote", out, "and", out.with_suffix(".pt"))
