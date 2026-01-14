"""Smoke tests for the DRCNN + AdaClip gradient training script.

These tests are intentionally light-weight and do not run a full training loop
on real data. Instead, they validate that:

- The hydra entrypoint can be imported.
- The `main` function can be called with a minimal synthetic config
  without raising obvious wiring / type errors up to early exit points.

The goal is to catch integration regressions in `examples/advanced/drcnn_adaclip_gradient_training.py`
without incurring the cost of end-to-end training in CI.
"""

from __future__ import annotations

import types

import pytest
from omegaconf import DictConfig, OmegaConf

from examples.advanced import drcnn_adaclip_gradient_training as drcnn_script


def _make_minimal_cfg() -> DictConfig:
    """Create a minimal DictConfig compatible with the script's expectations.

    We keep this configuration very small and synthetic and rely on the fact
    that the script uses Hydra defaults for most options.
    """

    # Minimal data section; values here are dummy and only need to be structurally valid.
    data = {
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
        # Any additional keys expected by SingleCu3sDataModule should be added here
        # if this test starts failing due to validation issues.
    }

    training = {
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "callbacks": None,
        },
        "optimizer": {
            "name": "adam",
            "lr": 1e-3,
        },
        "scheduler": None,
    }

    cfg_dict = {
        "name": "test_drcnn_adaclip",
        "output_dir": ".pytest_drcnn_output",
        "data": data,
        "training": training,
        # allow debug section but keep it disabled to avoid any I/O side effects
        "debug": {"save_intermediates": False},
        # unfreeze_nodes is optional in the script and defaults to [mixer.name]
    }

    return OmegaConf.create(cfg_dict)  # type: ignore[return-value]


def test_drcnn_script_imports_and_has_main() -> None:
    """The advanced DRCNN script should import and expose a callable `main`."""
    assert isinstance(drcnn_script, types.ModuleType)
    assert hasattr(drcnn_script, "main")
    assert callable(drcnn_script.main)


@pytest.mark.skip(
    reason="Integration-heavy: requires AdaCLIP + dataset; enable manually if needed."
)
def test_drcnn_script_main_smoke() -> None:
    """Optional smoke test for the DRCNN script main function.

    This test is skipped by default because it requires the external
    `cuvis_ai_adaclip` package and real data on disk. It can be enabled
    locally to validate integration end-to-end.
    """
    cfg = _make_minimal_cfg()

    # Ensure we pass a DictConfig instance as expected by the script.
    assert isinstance(cfg, DictConfig)

    # The script is expected to run without raising until it hits any genuine
    # runtime / wiring errors. Any such exception will fail this test.
    drcnn_script.main(cfg)  # type: ignore[arg-type]
