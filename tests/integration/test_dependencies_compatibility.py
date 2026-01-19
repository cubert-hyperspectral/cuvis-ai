from hydra import compose, initialize_config_dir  # type: ignore
from omegaconf import OmegaConf
from pydantic import BaseModel


def test_existing_code_still_imports():
    """Verify existing code can still import (with deprecations)."""
    from cuvis_ai_core.grpc.service import CuvisAIService  # noqa: F401
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline  # noqa: F401
    from cuvis_ai_core.training.config import TrainingConfig  # noqa: F401


def test_pydantic_hydra_integration(tmp_path):
    """Test Pydantic and Hydra work together."""

    class TestModel(BaseModel):
        value: int = 42

    cfg = OmegaConf.create({"value": 100})
    model = TestModel(**cfg)
    assert model.value == 100

    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("value: 7", encoding="utf-8")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        composed = compose(config_name="config")
        model_from_hydra = TestModel(**composed)

    assert model_from_hydra.value == 7
