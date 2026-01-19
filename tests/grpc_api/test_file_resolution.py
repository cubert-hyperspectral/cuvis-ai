import pytest

from cuvis_ai_core.grpc import helpers


def test_find_config_file_relative(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "training.yaml"
    config_file.write_text("seed: 1")

    resolved = helpers._find_config_file("training.yaml", [str(config_dir)])
    assert resolved == config_file.resolve()


def test_find_config_file_absolute(tmp_path):
    config_file = tmp_path / "data.yaml"
    config_file.write_text("value: 1")

    resolved = helpers._find_config_file(str(config_file), [str(tmp_path)])
    assert resolved == config_file


def test_find_weights_file(tmp_path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    weights_file = weights_dir / "model.pt"
    weights_file.write_bytes(b"weights")

    resolved = helpers.find_weights_file("model", [str(weights_dir)])
    assert resolved == weights_file.resolve()


def test_missing_weights_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        helpers.find_weights_file("missing", [str(tmp_path)])
