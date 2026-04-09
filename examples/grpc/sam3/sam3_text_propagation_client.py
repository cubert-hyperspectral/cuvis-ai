"""Run SAM3 text propagation via the gRPC Inference API.

Supports CU3S or RGB video sources and writes tracking results as video_coco JSON.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataset
from cuvis_ai_core.data.video import VideoFrameDataset, VideoIterator
from cuvis_ai_core.grpc import helpers
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cuvis_ai.node.json_writer import CocoTrackMaskWriter
from cuvis_ai.node.prompts import (
    load_text_prompt_schedule,
    resolve_text_prompt_for_frame,
)
from cuvis_ai.utils.grpc_workflow import (
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
    load_manifest_bytes,
    normalize_pipeline_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILTIN_PLUGINS_YAML = REPO_ROOT / "configs" / "plugins" / "cuvis_ai_builtin.yaml"
DEFAULT_CU3S_PIPELINE = "configs/pipeline/sam3/sam3_text_propagation.yaml"
DEFAULT_VIDEO_PIPELINE = "configs/pipeline/sam3/sam3_text_propagation_video.yaml"


def repo_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def pick(outputs: dict[str, cuvis_ai_pb2.Tensor], name: str) -> cuvis_ai_pb2.Tensor:
    return outputs.get(name) or next(
        tensor for key, tensor in outputs.items() if key.endswith(f".{name}")
    )


def to_numpy(outputs: dict, name: str, dtype) -> np.ndarray:
    return np.asarray(helpers.proto_to_numpy(pick(outputs, name)), dtype=dtype)


def _resolve_source_type(
    cu3s_path: Path | None,
    video_path: Path | None,
) -> str:
    if (cu3s_path is None) == (video_path is None):
        raise click.UsageError("Exactly one of --cu3s-path or --video-path must be provided.")
    return "cu3s" if cu3s_path is not None else "video"


def _resolve_pipeline_path(source_type: str, pipeline_path: str | None) -> str:
    if pipeline_path:
        return pipeline_path
    if source_type == "video":
        return DEFAULT_VIDEO_PIPELINE
    return DEFAULT_CU3S_PIPELINE


def _build_cu3s_loader(
    *,
    cu3s_path: Path,
    processing_mode: str,
    start_frame: int,
    max_frames: int | None,
) -> DataLoader:
    measurement_indices = (
        range(start_frame, start_frame + max_frames)
        if max_frames is not None and max_frames >= 0
        else range(start_frame, -1)
    )
    dataset = SingleCu3sDataset(
        cu3s_file_path=str(cu3s_path),
        processing_mode=processing_mode,
        measurement_indices=measurement_indices,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def _build_video_loader(
    *,
    video_path: Path,
    start_frame: int,
    max_frames: int | None,
) -> DataLoader:
    if max_frames == 0:
        raise click.ClickException("--max-frames=0 selects no frames.")

    end_frame = start_frame + max_frames if max_frames is not None and max_frames >= 0 else -1
    base_dataset = VideoFrameDataset(VideoIterator(str(video_path)), end_frame=end_frame)

    if start_frame >= len(base_dataset):
        raise click.ClickException(
            f"--start-frame={start_frame} is out of range for video length {len(base_dataset)}."
        )

    dataset: VideoFrameDataset | Subset
    if start_frame > 0:
        dataset = Subset(base_dataset, range(start_frame, len(base_dataset)))
    else:
        dataset = base_dataset

    if len(dataset) <= 0:
        raise click.ClickException("No frames selected for inference.")

    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def _build_inference_inputs(
    source_type: str,
    batch: dict,
    *,
    text_prompt: str | None = None,
) -> cuvis_ai_pb2.InputBatch:
    if source_type == "cu3s":
        request = cuvis_ai_pb2.InputBatch(
            cube=helpers.tensor_to_proto(batch["cube"]),
            wavelengths=helpers.tensor_to_proto(batch["wavelengths"]),
            mesu_index=helpers.tensor_to_proto(batch["mesu_index"]),
        )
    else:
        request = cuvis_ai_pb2.InputBatch(
            rgb_image=helpers.tensor_to_proto(batch["rgb_image"]),
            frame_id=helpers.tensor_to_proto(batch["frame_id"]),
        )
    if text_prompt:
        request.text_prompt = text_prompt
    return request


def run_client(
    *,
    cu3s_path: Path | None,
    video_path: Path | None,
    output_json_path: Path,
    server_address: str,
    pipeline_path: str | None,
    plugins_yaml: Path,
    processing_mode: str,
    start_frame: int,
    max_frames: int | None,
    prompt: tuple[str, ...],
) -> None:
    if start_frame < 0:
        raise click.BadParameter("--start-frame must be zero or positive.")
    if max_frames is not None and max_frames < -1:
        raise click.BadParameter("--max-frames must be -1 or non-negative.")

    source_type = _resolve_source_type(cu3s_path=cu3s_path, video_path=video_path)
    selected_pipeline_path = _resolve_pipeline_path(
        source_type=source_type,
        pipeline_path=pipeline_path,
    )
    prompt_schedule = load_text_prompt_schedule(prompt)

    if source_type == "video":
        print("Ignoring --processing-mode because --video-path is set.")

    builtin_plugins_yaml = BUILTIN_PLUGINS_YAML.resolve()
    if not builtin_plugins_yaml.exists():
        raise click.ClickException(f"Built-in plugin manifest not found: {builtin_plugins_yaml}")
    if not builtin_plugins_yaml.is_file():
        raise click.ClickException(
            f"Built-in plugin manifest is not a file: {builtin_plugins_yaml}"
        )

    plugins_yaml = repo_path(plugins_yaml)
    pipeline_file = repo_path(selected_pipeline_path)

    pipeline_request_path = str(pipeline_file) if pipeline_file.exists() else selected_pipeline_path
    search_paths = config_search_paths()
    if pipeline_file.exists():
        search_paths = [str(pipeline_file.parent), *search_paths]

    stub = build_stub(server_address=server_address, max_msg_size=600 * 1024 * 1024)
    session_id = create_session_with_search_paths(stub, search_paths)
    tracking_writer: CocoTrackMaskWriter | None = None
    print(f"Session: {session_id}")

    try:
        stub.LoadPlugins(
            cuvis_ai_pb2.LoadPluginsRequest(
                session_id=session_id,
                manifest=cuvis_ai_pb2.PluginManifest(
                    config_bytes=load_manifest_bytes(builtin_plugins_yaml)
                ),
            )
        )
        stub.LoadPlugins(
            cuvis_ai_pb2.LoadPluginsRequest(
                session_id=session_id,
                manifest=cuvis_ai_pb2.PluginManifest(
                    config_bytes=load_manifest_bytes(plugins_yaml)
                ),
            )
        )

        resolved = stub.ResolveConfig(
            cuvis_ai_pb2.ResolveConfigRequest(
                session_id=session_id,
                config_type="pipeline",
                path=pipeline_request_path,
            )
        )
        pipeline_config_bytes = normalize_pipeline_bytes(resolved.config_bytes)

        stub.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_config_bytes),
            )
        )

        effective_max_frames = None if max_frames is None or max_frames < 0 else max_frames

        if source_type == "cu3s":
            assert cu3s_path is not None
            loader = _build_cu3s_loader(
                cu3s_path=cu3s_path,
                processing_mode=processing_mode,
                start_frame=start_frame,
                max_frames=effective_max_frames,
            )
        else:
            assert video_path is not None
            loader = _build_video_loader(
                video_path=video_path,
                start_frame=start_frame,
                max_frames=effective_max_frames,
            )

        total_frames = len(loader.dataset)
        if total_frames <= 0:
            raise click.ClickException("No frames selected for inference.")

        tracking_writer = CocoTrackMaskWriter(
            output_json_path=output_json_path,
            default_category_name="object",
        )

        frame_id_key = "mesu_index" if source_type == "cu3s" else "frame_id"
        for batch in tqdm(loader, total=total_frames, desc="SAM3 gRPC", unit="frame"):
            if frame_id_key not in batch:
                raise click.ClickException(
                    f"Expected '{frame_id_key}' in input batch but key is missing."
                )
            frame_id = int(batch[frame_id_key].reshape(-1)[0].item())
            current_prompt = resolve_text_prompt_for_frame(
                prompt_schedule,
                frame_id,
                prompt_mode="scheduled",
            )

            response = stub.Inference(
                cuvis_ai_pb2.InferenceRequest(
                    session_id=session_id,
                    inputs=_build_inference_inputs(
                        source_type=source_type,
                        batch=batch,
                        text_prompt=current_prompt,
                    ),
                )
            )

            object_ids_t = torch.from_numpy(to_numpy(response.outputs, "object_ids", np.int64)).to(
                dtype=torch.int64
            )
            detection_scores_t = torch.from_numpy(
                to_numpy(response.outputs, "detection_scores", np.float32)
            ).to(dtype=torch.float32)
            if object_ids_t.numel() != detection_scores_t.numel():
                raise click.ClickException(
                    "SAM3 outputs are misaligned: "
                    f"object_ids={object_ids_t.numel()}, "
                    f"detection_scores={detection_scores_t.numel()}."
                )
            mask_t = torch.from_numpy(to_numpy(response.outputs, "mask", np.int32)).to(
                dtype=torch.int32
            )
            category_ids_t = torch.from_numpy(
                to_numpy(response.outputs, "category_ids", np.int64)
            ).to(dtype=torch.int64)
            category_semantics_t = torch.from_numpy(
                to_numpy(response.outputs, "category_semantics", np.uint8)
            ).to(dtype=torch.uint8)
            tracking_writer.forward(
                frame_id=torch.tensor([frame_id], dtype=torch.int64),
                mask=mask_t,
                object_ids=object_ids_t,
                detection_scores=detection_scores_t,
                category_ids=category_ids_t,
                category_semantics=category_semantics_t,
            )

        tracking_writer.close()
        tracking_writer = None
        print(f"Wrote tracking COCO JSON: {output_json_path}")

    finally:
        try:
            if tracking_writer is not None:
                tracking_writer.close()
        finally:
            stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


@click.command()
@click.option(
    "--cu3s-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to input .cu3s file.",
)
@click.option(
    "--video-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to input RGB video file.",
)
@click.option(
    "--output-json-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("outputs/sam3_grpc_tracking_results.json"),
    show_default=True,
    help="Output path for COCO tracking JSON.",
)
@click.option(
    "--server",
    "server_address",
    default="localhost:50051",
    show_default=True,
)
@click.option(
    "--pipeline-path",
    default=None,
    help=(
        "Optional pipeline path override. By default uses CU3S pipeline for "
        "--cu3s-path and video pipeline for --video-path."
    ),
)
@click.option(
    "--plugins-yaml",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("configs/plugins/sam3.yaml"),
    show_default=True,
)
@click.option(
    "--processing-mode",
    default="SpectralRadiance",
    show_default=True,
)
@click.option("--start-frame", type=int, default=0, show_default=True)
@click.option("--max-frames", type=int, default=-1, show_default=True)
@click.option(
    "--prompt",
    type=str,
    multiple=True,
    default=("person",),
    show_default=True,
    help=(
        "Repeatable text prompt spec: <text>@<frame_id>. Bare <text> means <text>@0. "
        "Prompts are emitted only on their scheduled frames."
    ),
)
def cli(
    cu3s_path: Path | None,
    video_path: Path | None,
    output_json_path: Path,
    server_address: str,
    pipeline_path: str | None,
    plugins_yaml: Path,
    processing_mode: str,
    start_frame: int,
    max_frames: int,
    prompt: tuple[str, ...],
) -> None:
    run_client(
        cu3s_path=cu3s_path,
        video_path=video_path,
        output_json_path=output_json_path,
        server_address=server_address,
        pipeline_path=pipeline_path,
        plugins_yaml=plugins_yaml,
        processing_mode=processing_mode,
        start_frame=start_frame,
        max_frames=max_frames,
        prompt=prompt,
    )


if __name__ == "__main__":
    cli()
