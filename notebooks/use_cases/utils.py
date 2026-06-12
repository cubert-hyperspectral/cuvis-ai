from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from cuvis_ai_dataloader.data import SingleCu3sDataModule
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from huggingface_hub import hf_hub_download
from IPython.display import Markdown, display
from loguru import logger

from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.utils.graph_helper import restructure_output_to_node_dict
from cuvis_ai_core.utils.node_registry import NodeRegistry

LENTILS_DEMO_REPO = "cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils"
DATA_SUBFOLDER = "measurements/cu3s/2026_04_15_13_32_55"

# Per-method asset locations on HuggingFace. Keys map to MethodSpec.name. Each
# entry points at the model-repo subfolder that holds the full pipeline YAML
# and the matching .pt weight. Methods whose subfolder does not yet exist on
# HF are skipped with a warning at runtime; see resolve_default_config().
METHOD_HF_ASSETS: dict[str, dict[str, str]] = {
    "dinomaly_rgb": {
        "subfolder": "dinomaly_rgb_full_pipeline",
        "yaml": "dinomaly_rgb.yaml",
        "pt": "dinomaly_rgb.pt",
    },
    "dinomaly_cir": {
        "subfolder": "dinomaly_cir_full_pipeline",
        "yaml": "dinomaly_cir.yaml",
        "pt": "dinomaly_cir.pt",
    },
    "dinomaly_concrete": {
        "subfolder": "dinomaly_custom_selector_full_pipeline",
        "yaml": "dinomaly_custom.yaml",
        "pt": "dinomaly_custom.pt",
    },
}


def _fetch_model(filename: str, *, subfolder: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=LENTILS_DEMO_REPO,
            subfolder=subfolder,
            filename=filename,
        )
    )


def _fetch_data(filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=LENTILS_DEMO_REPO,
            repo_type="dataset",
            subfolder=DATA_SUBFOLDER,
            filename=filename,
        )
    )


@dataclass(frozen=True)
class MethodSpec:
    """Per-method pipeline paths and TwoStageBinaryDecider gates from YAML.

    ``image_threshold`` applies to the mean of the top ``top_k_fraction`` of
    per-pixel scores (max across channels), matching the decider stage-1 gate.
    ``quantile`` is only for stage-2 pixel thresholding, not for the image gate.
    """

    name: str
    title: str
    yaml_path: Path
    pt_path: Path
    image_threshold: float
    quantile: float
    top_k_fraction: float


METHOD_TITLES = {
    "dinomaly_rgb": "Dinomaly RGB",
    "dinomaly_cir": "Dinomaly CIR",
    "dinomaly_concrete": "Dinomaly Custom Selector",
}


def read_two_stage_decider_hparams(yaml_path: Path) -> tuple[float, float, float]:
    """Read ``image_threshold``, ``quantile``, and ``top_k_fraction`` from the decider node."""
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    for node in data.get("nodes", []):
        if str(node.get("name", "")) != "decider":
            continue
        hp = node.get("hparams") or {}
        if "image_threshold" not in hp or "quantile" not in hp:
            continue
        t = float(hp["image_threshold"])
        q = float(hp["quantile"])
        tkf = float(hp.get("top_k_fraction", 0.001))
        return t, q, tkf
    msg = f"No decider hparams with image_threshold and quantile in {yaml_path}"
    raise ValueError(msg)


def read_two_stage_decider_thresholds(yaml_path: Path) -> tuple[float, float]:
    """Read ``image_threshold`` and ``quantile`` only (backward-compatible wrapper)."""
    t, q, _ = read_two_stage_decider_hparams(yaml_path)
    return t, q


def _method_spec_from_yaml(*, name: str, title: str, yaml_path: Path, pt_path: Path) -> MethodSpec:
    t, q, tkf = read_two_stage_decider_hparams(yaml_path)
    return MethodSpec(
        name=name,
        title=title,
        yaml_path=yaml_path,
        pt_path=pt_path,
        image_threshold=t,
        quantile=q,
        top_k_fraction=tkf,
    )


def resolve_default_config() -> dict[str, Any]:
    """Resolve tutorial assets, fetching pipelines + sample CU3S from HuggingFace.

    The custom-selector flow is the only method with a complete pipeline on HF
    today. RGB and CIR are attempted but skipped with a warning if their
    HF subfolders do not exist yet.
    """
    use_cases = Path(__file__).resolve().parent
    repo_root = use_cases.parent.parent

    methods: list[MethodSpec] = []
    for name, assets in METHOD_HF_ASSETS.items():
        try:
            yaml_path = _fetch_model(assets["yaml"], subfolder=assets["subfolder"])
            pt_path = _fetch_model(assets["pt"], subfolder=assets["subfolder"])
        except Exception as exc:  # noqa: BLE001 - HF returns several exception types
            logger.warning(
                f"Skipping method '{name}': could not fetch assets from "
                f"{LENTILS_DEMO_REPO!r}/{assets['subfolder']!r} "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        methods.append(
            _method_spec_from_yaml(
                name=name,
                title=METHOD_TITLES[name],
                yaml_path=yaml_path,
                pt_path=pt_path,
            )
        )

    if not methods:
        raise RuntimeError(
            "No method assets could be fetched from HuggingFace. Check that "
            f"{LENTILS_DEMO_REPO!r} is reachable and at least one *_full_pipeline "
            "subfolder exists."
        )

    cu3s_path = _fetch_data("Auto_003+01.cu3s")
    _fetch_data("Auto_003+01.info")
    annotation_json_path = _fetch_data("Auto_003+01.json")

    return {
        "cu3s_path": cu3s_path,
        "annotation_json_path": annotation_json_path,
        "plugins_manifest": repo_root / "configs" / "plugins" / "dinomaly.yaml",
        "methods": methods,
    }


def assert_paths_exist(config: dict[str, Any]) -> None:
    """Verify locally-resolvable paths in *config* exist.

    HF-fetched paths are validated by ``hf_hub_download`` itself; we only
    re-check the ones we resolve from the repo (the plugin manifest) and any
    paths the caller may have overridden.
    """
    required = [
        config["cu3s_path"],
        config["annotation_json_path"],
        config["plugins_manifest"],
    ]
    for method in config["methods"]:
        required.extend([method.yaml_path, method.pt_path])
    missing = [str(p) for p in required if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def make_predict_loader(
    cu3s_path: Path,
    annotation_json_path: Path,
    *,
    processing_mode: str = "Reflectance",
    batch_size: int = 1,
) -> tuple[SingleCu3sDataModule, Any]:
    dm = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        annotation_json_path=str(annotation_json_path),
        train_ids=[],
        val_ids=[],
        test_ids=[],
        predict_ids=None,
        batch_size=batch_size,
        processing_mode=processing_mode,
        normalize_to_unit=False,
    )
    dm.setup(stage="predict")
    return dm, dm.predict_dataloader()


def load_pipeline_for_inference(
    *,
    yaml_path: Path,
    pt_path: Path,
    plugins_manifest: Path,
    device: str,
) -> CuvisPipeline:
    registry = NodeRegistry()
    registry.register_plugins(str(plugins_manifest))
    pipeline = CuvisPipeline.load_pipeline(
        yaml_path,
        weights_path=str(pt_path),
        device=device,
        strict_weight_loading=False,
        node_registry=registry,
    )
    pipeline.torch_layers.eval()
    return pipeline


def _to_numpy(x: Any) -> np.ndarray | None:
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return None


def _pick_node_payload(node_out: dict[str, Any], key: str) -> dict[str, Any]:
    preferred = {
        "scores": ("dinomaly_detector", "DinomalyDetector"),
        "decisions": ("decider", "quantile_decider", "QuantileBinaryDecider"),
        "rgb_image": ("rgb_selector", "cir_selector", "FixedWavelengthSelector", "CIRSelector"),
    }
    for name in preferred.get(key, ()):
        payload = node_out.get(name)
        if isinstance(payload, dict) and payload.get(key) is not None:
            return payload
    for payload in node_out.values():
        if isinstance(payload, dict) and payload.get(key) is not None:
            return payload
    return {}


def _sample_tensor_value(x: Any, idx: int, expected_batch: int) -> np.ndarray | None:
    if not torch.is_tensor(x):
        return None
    if x.ndim == 0:
        return _to_numpy(x) if idx == 0 else None
    if x.shape[0] == expected_batch:
        return _to_numpy(x[idx])
    return _to_numpy(x) if idx == 0 else None


def _as_bool_mask(mask: np.ndarray, normal_class_ids: set[int]) -> np.ndarray:
    return ~np.isin(mask.astype(np.int32, copy=False), list(normal_class_ids))


def _image_score_topk_mean(anomaly_map: np.ndarray, top_k_fraction: float) -> float:
    """Image-level score matching ``TwoStageBinaryDecider`` stage-1 (mean of top-k pixel scores)."""
    x = anomaly_map.astype(np.float64, copy=False)
    if x.ndim == 3:
        pixel_scores = np.max(x, axis=-1)
    else:
        pixel_scores = x
    flat = pixel_scores.ravel()
    n = int(flat.size)
    if n == 0:
        return 0.0
    k = int(np.ceil(np.float32(n) * np.float32(top_k_fraction)))
    k = max(1, min(k, n))
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].mean())


def _pixel_gate(anomaly_map_hw: np.ndarray, quantile: float) -> tuple[np.ndarray, float]:
    thr = float(np.quantile(anomaly_map_hw.astype(np.float64).ravel(), quantile))
    return anomaly_map_hw >= thr, thr


def run_method_on_frame_subset(
    *,
    pipeline: CuvisPipeline,
    loader: Any,
    frame_indices: set[int],
    image_threshold: float,
    quantile: float,
    top_k_fraction: float,
    device: torch.device,
    normal_class_ids: set[int],
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    running_index = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            bsz = int(batch["cube"].shape[0])
            keep_local = [i for i in range(bsz) if (running_index + i) in frame_indices]
            if not keep_local:
                running_index += bsz
                continue
            context = Context(
                stage=ExecutionStage.TEST,
                epoch=0,
                batch_idx=batch_idx,
                global_step=running_index,
            )
            batch_dev = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            raw = pipeline.forward(batch_dev, context=context)
            node_out = restructure_output_to_node_dict(raw)
            det = _pick_node_payload(node_out, "scores")
            dec = _pick_node_payload(node_out, "decisions")
            sel = _pick_node_payload(node_out, "rgb_image")

            for i in keep_local:
                abs_idx = running_index + i
                scores = _sample_tensor_value(det.get("scores"), i, bsz)
                if scores is None:
                    continue
                anomaly_map = scores
                if anomaly_map.ndim == 3 and anomaly_map.shape[-1] == 1:
                    anomaly_map = anomaly_map[..., 0]
                pred_px, px_thr = _pixel_gate(anomaly_map, quantile=quantile)
                image_score = _image_score_topk_mean(anomaly_map, top_k_fraction=top_k_fraction)
                if image_score < image_threshold:
                    pred_px = np.zeros_like(pred_px, dtype=bool)

                pred_from_node = _sample_tensor_value(dec.get("decisions"), i, bsz)
                if pred_from_node is not None:
                    if pred_from_node.ndim == 3 and pred_from_node.shape[-1] == 1:
                        pred_from_node = pred_from_node[..., 0]
                    pred_node_bool = pred_from_node.astype(bool)
                else:
                    pred_node_bool = pred_px

                rgb = _sample_tensor_value(sel.get("rgb_image"), i, bsz)
                mask = _sample_tensor_value(batch.get("mask"), i, bsz)
                if mask is None:
                    continue
                gt_bool = _as_bool_mask(mask, normal_class_ids=normal_class_ids)

                if rgb is None:
                    rgb = np.zeros((*anomaly_map.shape, 3), dtype=np.float32)

                out_rows.append(
                    {
                        "frame_idx": abs_idx,
                        "mesu_index": int(batch["mesu_index"][i].item()),
                        "rgb_image": rgb.astype(np.float32),
                        "anomaly_map": anomaly_map.astype(np.float32),
                        "pred_mask": pred_node_bool.astype(bool),
                        "pred_mask_quantile": pred_px.astype(bool),
                        "gt_mask": gt_bool.astype(bool),
                        "image_score": float(image_score),
                        "pixel_threshold": float(px_thr),
                    }
                )
            running_index += bsz
    return out_rows


def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    if x.max() <= 1.0:
        x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    return x


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def overlay_boundaries(
    rgb_u8: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray
) -> np.ndarray:
    out = rgb_u8.copy()
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    both = np.logical_and(pred, gt)
    out[gt] = np.array([0, 255, 0], dtype=np.uint8)
    out[pred] = np.array([255, 0, 0], dtype=np.uint8)
    out[both] = np.array([255, 255, 0], dtype=np.uint8)
    return out


def render_method_triplet(row: dict[str, Any], *, title: str) -> None:
    rgb = to_uint8_rgb(row["rgb_image"])
    amap = row["anomaly_map"]
    pred = row["pred_mask"]
    gt = row["gt_mask"]
    overlay = overlay_boundaries(rgb, pred, gt)
    score = row["image_score"]
    metric_iou = iou(pred, gt)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=140)
    axes[0].imshow(rgb)
    axes[0].set_title("Input")
    axes[1].imshow(amap, cmap="inferno")
    axes[1].set_title(f"Anomaly map\nscore={score:.4f}")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Prediction")
    axes[3].imshow(overlay)
    axes[3].set_title(f"Overlay (IoU={metric_iou:.3f})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()


def render_input_triplet(rows_by_method: dict[str, list[dict[str, Any]]], frame_idx: int) -> None:
    """Show input view differences (RGB/CIR/Custom) for one frame."""
    entries: list[tuple[str, np.ndarray]] = []
    for key in ("dinomaly_rgb", "dinomaly_cir", "dinomaly_concrete"):
        rows = rows_by_method.get(key, [])
        row = next((r for r in rows if int(r["frame_idx"]) == int(frame_idx)), None)
        if row is None:
            continue
        entries.append((METHOD_TITLES[key], to_uint8_rgb(row["rgb_image"])))
    if not entries:
        raise ValueError(f"No rows found for frame_idx={frame_idx}")

    fig, axes = plt.subplots(1, len(entries), figsize=(5.5 * len(entries), 4), dpi=140)
    if len(entries) == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, entries, strict=True):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Input view comparison | frame_idx={frame_idx}")
    plt.tight_layout()


def summarize_subset(
    rows_by_method: dict[str, list[dict[str, Any]]],
) -> list[dict[str, float | str | int]]:
    table: list[dict[str, float | str | int]] = []
    for method_name in ("dinomaly_rgb", "dinomaly_cir", "dinomaly_concrete"):
        rows = rows_by_method.get(method_name, [])
        if not rows:
            continue
        table.append(
            {
                "method": METHOD_TITLES[method_name],
                "n_frames": int(len(rows)),
                "mean_iou": float(np.mean([iou(r["pred_mask"], r["gt_mask"]) for r in rows])),
                "mean_image_score": float(np.mean([float(r["image_score"]) for r in rows])),
                "mean_pixel_threshold": float(np.mean([float(r["pixel_threshold"]) for r in rows])),
            }
        )
    return table


def _overlay_frame_from_row(row: dict[str, Any]) -> np.ndarray:
    return overlay_boundaries(to_uint8_rgb(row["rgb_image"]), row["pred_mask"], row["gt_mask"])


def export_overlay_video(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    fps: float = 4.0,
    overlay_title: str | None = None,
) -> Path:
    """Export per-frame overlays as H.264 MP4 using imageio-ffmpeg."""
    if not rows:
        raise ValueError("No rows to export")
    rows_sorted = sorted(rows, key=lambda r: int(r["frame_idx"]))
    output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path),
        fps=float(fps),
        codec="libx264",
        quality=5,
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-movflags", "+faststart"],
    ) as writer:
        for row in rows_sorted:
            writer.append_data(_overlay_frame_from_row(row))
    return output_path


def _format_table_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: Iterable[str]) -> str:
    columns = list(columns)
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = [_format_table_value(row.get(col, "")) for col in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def display_records_table(
    rows: list[dict[str, Any]], columns: Iterable[str], *, title: str | None = None
) -> None:
    """Render *rows* as a Markdown table in a Jupyter cell output."""
    markdown = _markdown_table(rows, columns)
    if title:
        markdown = f"**{title}**\n\n{markdown}"
    display(Markdown(markdown))
