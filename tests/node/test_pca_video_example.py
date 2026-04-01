from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import pytest
from cuvis_ai_schemas.enums import ExecutionStage

import examples.blood_perfusion.pca as export_mod


class _FakeDataModule:
    created: list[_FakeDataModule] = []
    predict_dataset: object | None = None

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.train_ids = kwargs.get("train_ids")
        self.predict_ids = kwargs.get("predict_ids")
        self.predict_ds = type(self).predict_dataset
        type(self).created.append(self)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        return None


class _FakeCU3SDataNode:
    def __init__(self, **_: object) -> None:
        self.outputs = SimpleNamespace(cube="cube_out", mesu_index="mesu_out")


class _FakePCA:
    created: list[_FakePCA] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.data = "pca_data_in"
        self.outputs = SimpleNamespace(
            projected="pca_projected_out",
            explained_variance_ratio="pca_ratio_out",
            components="pca_components_out",
        )
        type(self).created.append(self)


class _FakeTrainablePCA(_FakePCA):
    created: list[_FakeTrainablePCA] = []


class _FakeExplainedVarianceMetric:
    created: list[_FakeExplainedVarianceMetric] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.explained_variance_ratio = "metric_ratio_in"
        self.outputs = SimpleNamespace(metrics="metrics_out")
        type(self).created.append(self)


class _FakeMinMaxNormalizer:
    def __init__(self, **_: object) -> None:
        self.data = "normalizer_data_in"
        self.normalized = "normalizer_out"


class _FakeToVideoNode:
    output_video_path: str = ""

    def __init__(self, **kwargs: object) -> None:
        type(self).output_video_path = str(kwargs["output_video_path"])
        self.rgb_image = "rgb_in"
        self.frame_id = "frame_id_in"


class _FakeMonochromeToRGBNode:
    created: list[_FakeMonochromeToRGBNode] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.data = "mono_in"
        self.rgb_image = "mono_rgb_out"
        type(self).created.append(self)


class _FakePipeline:
    created: list[_FakePipeline] = []

    def __init__(self, name: str) -> None:
        self.name = name
        self.connections: list[tuple[object, object]] = []
        type(self).created.append(self)

    def connect(self, *connections: tuple[object, object]) -> None:
        self.connections.extend(connections)

    def visualize(self, **_: object) -> None:
        return None

    def to(self, _: object) -> None:
        return None

    def save_to_file(self, path: str) -> None:
        Path(path).write_text("pipeline: fake\n", encoding="utf-8")


class _FakeStatisticalTrainer:
    created: list[_FakeStatisticalTrainer] = []

    def __init__(self, pipeline: object, datamodule: object) -> None:
        self.pipeline = pipeline
        self.datamodule = datamodule
        self.fit_calls = 0
        type(self).created.append(self)

    def fit(self) -> None:
        self.fit_calls += 1


class _FakePredictor:
    created: list[_FakePredictor] = []

    def __init__(self, pipeline: object, datamodule: object) -> None:
        self.pipeline = pipeline
        self.datamodule = datamodule
        self.max_batches: int | None = None
        self.collect_outputs: bool | None = None
        type(self).created.append(self)

    def predict(self, max_batches: int | None = None, collect_outputs: bool = False) -> None:
        self.max_batches = max_batches
        self.collect_outputs = collect_outputs
        Path(_FakeToVideoNode.output_video_path).write_bytes(b"fake")


class _FakeLogger:
    def __init__(self) -> None:
        self.add_calls: list[tuple[Path, dict[str, object]]] = []
        self.remove_calls: list[int] = []
        self.messages: list[tuple[str, str, tuple[object, ...]]] = []
        self._next_sink_id = 1

    def add(self, sink: str | Path, **kwargs: object) -> int:
        sink_path = Path(sink)
        sink_path.parent.mkdir(parents=True, exist_ok=True)
        sink_path.write_text("", encoding="utf-8")
        self.add_calls.append((sink_path, kwargs))
        sink_id = self._next_sink_id
        self._next_sink_id += 1
        return sink_id

    def remove(self, sink_id: int) -> None:
        self.remove_calls.append(sink_id)

    def info(self, message: str, *args: object) -> None:
        self.messages.append(("info", message, args))

    def warning(self, message: str, *args: object) -> None:
        self.messages.append(("warning", message, args))

    def success(self, message: str, *args: object) -> None:
        self.messages.append(("success", message, args))


@pytest.fixture
def pca_export_context(
    monkeypatch: pytest.MonkeyPatch,
    create_test_predict_dataset,
) -> dict[str, Any]:
    _FakeDataModule.created.clear()
    _FakeDataModule.predict_dataset = create_test_predict_dataset(
        total_frames=6,
        height=2,
        width=2,
        num_channels=5,
    )
    _FakePCA.created.clear()
    _FakeTrainablePCA.created.clear()
    _FakeExplainedVarianceMetric.created.clear()
    _FakeMonochromeToRGBNode.created.clear()
    _FakePipeline.created.clear()
    _FakeStatisticalTrainer.created.clear()
    _FakePredictor.created.clear()
    fake_logger = _FakeLogger()

    monkeypatch.setattr(export_mod, "SingleCu3sDataModule", _FakeDataModule)
    monkeypatch.setattr(export_mod, "CU3SDataNode", _FakeCU3SDataNode)
    monkeypatch.setattr(export_mod, "PCA", _FakePCA)
    monkeypatch.setattr(export_mod, "TrainablePCA", _FakeTrainablePCA)
    monkeypatch.setattr(export_mod, "ExplainedVarianceMetric", _FakeExplainedVarianceMetric)
    monkeypatch.setattr(export_mod, "MinMaxNormalizer", _FakeMinMaxNormalizer)
    monkeypatch.setattr(export_mod, "MonochromeToRGBNode", _FakeMonochromeToRGBNode)
    monkeypatch.setattr(export_mod, "ToVideoNode", _FakeToVideoNode)
    monkeypatch.setattr(export_mod, "CuvisPipeline", _FakePipeline)
    monkeypatch.setattr(export_mod, "StatisticalTrainer", _FakeStatisticalTrainer)
    monkeypatch.setattr(export_mod, "Predictor", _FakePredictor)
    monkeypatch.setattr(export_mod, "logger", fake_logger)
    return {
        "datamodule_cls": _FakeDataModule,
        "pca_cls": _FakePCA,
        "trainable_pca_cls": _FakeTrainablePCA,
        "metric_cls": _FakeExplainedVarianceMetric,
        "monochrome_cls": _FakeMonochromeToRGBNode,
        "pipeline_cls": _FakePipeline,
        "trainer_cls": _FakeStatisticalTrainer,
        "predictor_cls": _FakePredictor,
        "logger": fake_logger,
    }


def test_export_global_mode_uses_trainable_pca_and_statistical_trainer(
    pca_export_context: dict[str, Any],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "global.mp4"

    result = export_mod.export_pca_video(
        cu3s_file_path="dummy.cu3s",
        output_video_path=str(output_path),
        pca_mode="global",
        processing_mode="Raw",
        start_frame=1,
        end_frame=4,
    )

    export_datamodule = pca_export_context["datamodule_cls"].created[-1]
    pipeline = pca_export_context["pipeline_cls"].created[-1]
    trainer = pca_export_context["trainer_cls"].created[-1]
    predictor = pca_export_context["predictor_cls"].created[-1]

    assert result == output_path
    assert output_path.exists()
    assert len(pca_export_context["trainable_pca_cls"].created) == 1
    assert len(pca_export_context["pca_cls"].created) == 0
    assert pca_export_context["trainable_pca_cls"].created[0].kwargs["n_components"] == 1
    assert trainer.fit_calls == 1
    assert export_datamodule.train_ids == [1, 2, 3]
    assert export_datamodule.predict_ids == [1, 2, 3]
    assert predictor.max_batches == 3
    assert predictor.collect_outputs is False
    assert ("pca_ratio_out", "metric_ratio_in") in pipeline.connections
    assert ("normalizer_out", "mono_in") in pipeline.connections
    assert ("mono_rgb_out", "rgb_in") in pipeline.connections
    assert pca_export_context["metric_cls"].created[0].kwargs["execution_stages"] == {
        ExecutionStage.ALWAYS
    }
    assert pca_export_context["logger"].add_calls == [
        (output_path.with_suffix(".log"), {"level": "INFO", "mode": "w"})
    ]
    assert output_path.with_suffix(".log").exists()
    assert pca_export_context["logger"].remove_calls == [1]


def test_export_global_mode_can_limit_statistical_init_frames(
    pca_export_context: dict[str, Any],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "global_init_split.mp4"

    result = export_mod.export_pca_video(
        cu3s_file_path="dummy.cu3s",
        output_video_path=str(output_path),
        pca_mode="global",
        processing_mode="Raw",
        start_frame=0,
        end_frame=-1,
        init_frames=2,
    )

    export_datamodule = pca_export_context["datamodule_cls"].created[-1]
    trainer = pca_export_context["trainer_cls"].created[-1]
    predictor = pca_export_context["predictor_cls"].created[-1]

    assert result == output_path
    assert output_path.exists()
    assert pca_export_context["trainable_pca_cls"].created[0].kwargs["n_components"] == 1
    assert trainer.fit_calls == 1
    assert export_datamodule.train_ids == [0, 1]
    assert export_datamodule.predict_ids is None
    assert predictor.max_batches == 6


def test_export_per_frame_mode_uses_library_pca_and_noop_statistical_fit(
    pca_export_context: dict[str, Any],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "per_frame.mp4"

    result = export_mod.export_pca_video(
        cu3s_file_path="dummy.cu3s",
        output_video_path=str(output_path),
        pca_mode="per_frame",
        processing_mode="Raw",
        start_frame=0,
        end_frame=-1,
    )

    export_datamodule = pca_export_context["datamodule_cls"].created[-1]
    trainer = pca_export_context["trainer_cls"].created[-1]
    predictor = pca_export_context["predictor_cls"].created[-1]

    assert result == output_path
    assert output_path.exists()
    assert len(pca_export_context["pca_cls"].created) == 1
    assert len(pca_export_context["trainable_pca_cls"].created) == 0
    assert pca_export_context["pca_cls"].created[0].kwargs["n_components"] == 1
    assert trainer.fit_calls == 1
    assert export_datamodule.train_ids == [0, 1, 2, 3, 4, 5]
    assert export_datamodule.predict_ids is None
    assert predictor.max_batches == 6


def test_export_rejects_non_positive_init_frames(
    pca_export_context: dict[str, Any],
    tmp_path: Path,
) -> None:
    del pca_export_context

    with pytest.raises(click.BadParameter, match="init-frames"):
        export_mod.export_pca_video(
            cu3s_file_path="dummy.cu3s",
            output_video_path=str(tmp_path / "bad.mp4"),
            pca_mode="global",
            processing_mode="Raw",
            init_frames=0,
        )
