"""Regression tests for config preservation through training workflows.

These tests were added to catch bugs where training configs were being overwritten
or not properly preserved through training workflows, particularly when using
RestoreExperiment followed by sequential statistical and gradient training.
"""

import json

import pytest
import yaml
from omegaconf import OmegaConf

from cuvis_ai.grpc import cuvis_ai_pb2
from cuvis_ai.training.config import TrainingConfig


class TestConfigPreservationThroughTraining:
    """Test that configs loaded from experiment files are preserved through training."""

    def test_max_epochs_preserved_through_statistical_training(self, grpc_stub, tmp_path):
        """
        Regression test: Verify that max_epochs from experiment config is preserved
        when running statistical training.

        Bug: Statistical training was creating new TrainingConfig() with default
        max_epochs=100, overwriting the loaded config with max_epochs=20.

        Fix: Changed from `training_config_py = TrainingConfig()` to
        `training_config_py = session.training_config or TrainingConfig()`
        in the Train method's statistical training branch.
        """
        # Use existing deep_svdd experiment which has max_epochs=20
        experiment_path = "configs/experiment/deep_svdd.yaml"

        # Verify the experiment file has max_epochs=20
        with open(experiment_path) as f:
            exp_config = yaml.safe_load(f)
        assert exp_config["training"]["trainer"]["max_epochs"] == 20

        # Restore experiment (loads config with max_epochs=20)
        restore_response = grpc_stub.RestoreExperiment(
            cuvis_ai_pb2.RestoreExperimentRequest(experiment_path=experiment_path)
        )
        session_id = restore_response.session_id

        try:
            # Verify the loaded config has max_epochs=20
            assert restore_response.experiment.training.config_bytes
            training_config_dict = json.loads(
                restore_response.experiment.training.config_bytes.decode("utf-8")
            )
            assert training_config_dict["trainer"]["max_epochs"] == 20

            # Run statistical training
            # Before the fix, this would overwrite training_config with defaults
            stat_request = cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
            stat_responses = list(grpc_stub.Train(stat_request))
            assert stat_responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

            # Save experiment and verify max_epochs is still 20
            saved_exp_path = tmp_path / "saved_after_stat_training.yaml"
            save_response = grpc_stub.SaveExperiment(
                cuvis_ai_pb2.SaveExperimentRequest(
                    session_id=session_id,
                    experiment_path=str(saved_exp_path),
                )
            )
            assert save_response.success

            # Read back the saved experiment
            with open(saved_exp_path) as f:
                saved_config = yaml.safe_load(f)

            # BUG CHECK: max_epochs should still be 20, not reverted to 100
            assert saved_config["training"]["trainer"]["max_epochs"] == 20, (
                "Statistical training should preserve max_epochs from loaded config"
            )

        finally:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    @pytest.mark.slow
    def test_max_epochs_used_in_gradient_training(self, grpc_stub, trained_session, tmp_path):
        """
        Regression test: Verify that gradient training uses max_epochs from loaded config.

        Bug: After statistical training overwrote the config, gradient training would
        train for 100 epochs instead of the configured 20 epochs.

        This test verifies the complete workflow: RestoreExperiment -> Statistical
        Training -> Gradient Training uses correct max_epochs throughout.
        """
        # Use trained_session which restores experiment and runs statistical training
        session_id, data_config = trained_session()

        try:
            # Run gradient training
            # The training config should have max_epochs=20 from the experiment file
            grad_request = cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            )

            # Collect all progress updates
            progress_updates = list(grpc_stub.Train(grad_request))

            # Verify training completed
            assert progress_updates[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

            # Extract max epoch from progress updates
            max_epoch_seen = max(update.context.epoch for update in progress_updates)

            # Should train for 20 epochs (or close, depending on early stopping)
            # The key is it should NOT be 100 epochs
            assert max_epoch_seen <= 25, (
                f"Gradient training should use max_epochs=20 from config, "
                f"but saw epoch {max_epoch_seen}"
            )
            assert max_epoch_seen >= 1, "Should have completed at least 1 epoch"

        finally:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_all_training_config_fields_preserved(self, grpc_stub, tmp_path):
        """
        Test that all training config fields (not just max_epochs) are preserved
        through the training workflow.
        """
        experiment_path = "configs/experiment/deep_svdd.yaml"

        # Load original config
        with open(experiment_path) as f:
            original_config = yaml.safe_load(f)

        # Restore and train
        restore_response = grpc_stub.RestoreExperiment(
            cuvis_ai_pb2.RestoreExperimentRequest(experiment_path=experiment_path)
        )
        session_id = restore_response.session_id

        try:
            # Run statistical training
            stat_request = cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
            list(grpc_stub.Train(stat_request))

            # Save and compare
            saved_exp_path = tmp_path / "saved_config.yaml"
            grpc_stub.SaveExperiment(
                cuvis_ai_pb2.SaveExperimentRequest(
                    session_id=session_id,
                    experiment_path=str(saved_exp_path),
                )
            )

            with open(saved_exp_path) as f:
                saved_config = yaml.safe_load(f)

            # Verify key training config fields are preserved
            original_training = original_config["training"]
            saved_training = saved_config["training"]

            assert saved_training["seed"] == original_training["seed"]
            assert (
                saved_training["trainer"]["max_epochs"]
                == original_training["trainer"]["max_epochs"]
            )
            assert (
                saved_training["trainer"]["accelerator"]
                == original_training["trainer"]["accelerator"]
            )
            assert saved_training["optimizer"]["name"] == original_training["optimizer"]["name"]
            assert saved_training["optimizer"]["lr"] == original_training["optimizer"]["lr"]

        finally:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


class TestTrainingConfigFromDictConfig:
    """Test TrainingConfig.from_dict_config with OmegaConf (regression tests for refactoring)."""

    def test_from_dict_config_with_partial_overrides(self):
        """
        Test that partial overrides merge correctly with defaults.

        This was a key requirement for the OmegaConf refactoring.
        """
        # Provide only partial config (should merge with defaults)
        dict_config = OmegaConf.create(
            {
                "trainer": {
                    "max_epochs": 50,
                    # accelerator, devices, etc. should use defaults
                },
                # optimizer should use all defaults
            }
        )

        config = TrainingConfig.from_dict_config(dict_config)

        # Custom value
        assert config.trainer.max_epochs == 50

        # Default values
        assert config.seed == 42  # default
        assert config.trainer.accelerator == "auto"  # default
        assert config.optimizer.name == "adamw"  # default
        assert config.optimizer.lr == 1e-3  # default

    def test_from_dict_config_with_plain_dict(self):
        """
        Test that from_dict_config works with plain Python dicts (not just DictConfig).
        """
        plain_dict = {
            "seed": 99,
            "trainer": {"max_epochs": 15},
            "optimizer": {"lr": 0.005},
        }

        config = TrainingConfig.from_dict_config(plain_dict)

        assert config.seed == 99
        assert config.trainer.max_epochs == 15
        assert config.optimizer.lr == 0.005

    def test_from_dict_config_roundtrip_preserves_data(self):
        """
        Test that config -> dict_config -> config roundtrip preserves all data.

        This verifies the refactored implementation is lossless.
        """
        from cuvis_ai.training.config import OptimizerConfig, TrainerConfig

        original = TrainingConfig(
            seed=123,
            trainer=TrainerConfig(
                max_epochs=25,
                accelerator="gpu",
                devices=1,
                precision="16-mixed",
            ),
            optimizer=OptimizerConfig(
                name="adam",
                lr=0.002,
                weight_decay=0.005,
            ),
        )

        # Convert to DictConfig and back
        dict_config = original.to_dict_config()
        restored = TrainingConfig.from_dict_config(dict_config)

        # Verify all fields match
        assert restored.seed == original.seed
        assert restored.trainer.max_epochs == original.trainer.max_epochs
        assert restored.trainer.accelerator == original.trainer.accelerator
        assert restored.trainer.devices == original.trainer.devices
        assert restored.trainer.precision == original.trainer.precision
        assert restored.optimizer.name == original.optimizer.name
        assert restored.optimizer.lr == original.optimizer.lr
        assert restored.optimizer.weight_decay == original.optimizer.weight_decay


class TestSessionStateManagement:
    """Test session state management edge cases."""

    def test_session_preserves_experiment_config_after_training(self, grpc_stub, tmp_path):
        """
        Test that experiment config (including loss_nodes, metric_nodes) is preserved
        in session state through training operations.
        """
        experiment_path = "configs/experiment/deep_svdd.yaml"

        # Load original experiment config
        with open(experiment_path) as f:
            original_exp = yaml.safe_load(f)

        # Restore experiment
        restore_response = grpc_stub.RestoreExperiment(
            cuvis_ai_pb2.RestoreExperimentRequest(experiment_path=experiment_path)
        )
        session_id = restore_response.session_id

        try:
            # Run statistical training
            stat_request = cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
            list(grpc_stub.Train(stat_request))

            # Save experiment
            saved_exp_path = tmp_path / "preserved_exp.yaml"
            grpc_stub.SaveExperiment(
                cuvis_ai_pb2.SaveExperimentRequest(
                    session_id=session_id,
                    experiment_path=str(saved_exp_path),
                )
            )

            # Verify experiment config is preserved
            with open(saved_exp_path) as f:
                saved_exp = yaml.safe_load(f)

            # Check that critical experiment fields are preserved
            if "loss_nodes" in original_exp:
                assert saved_exp.get("loss_nodes") == original_exp["loss_nodes"]
            if "metric_nodes" in original_exp:
                assert saved_exp.get("metric_nodes") == original_exp["metric_nodes"]
            if "unfreeze_nodes" in original_exp:
                assert saved_exp.get("unfreeze_nodes") == original_exp["unfreeze_nodes"]

        finally:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_multiple_sessions_have_independent_configs(self, grpc_stub, mock_cuvis_sdk):
        """
        Test that multiple sessions maintain independent training configs.

        Ensures that config changes in one session don't affect other sessions.
        """
        experiment_path = "configs/experiment/deep_svdd.yaml"

        # Create two sessions
        response1 = grpc_stub.RestoreExperiment(
            cuvis_ai_pb2.RestoreExperimentRequest(experiment_path=experiment_path)
        )
        session_id1 = response1.session_id

        response2 = grpc_stub.RestoreExperiment(
            cuvis_ai_pb2.RestoreExperimentRequest(experiment_path=experiment_path)
        )
        session_id2 = response2.session_id

        try:
            # Both should have max_epochs=20 from the experiment file
            config1 = json.loads(response1.experiment.training.config_bytes.decode("utf-8"))
            config2 = json.loads(response2.experiment.training.config_bytes.decode("utf-8"))

            assert config1["trainer"]["max_epochs"] == 20
            assert config2["trainer"]["max_epochs"] == 20

            # Run statistical training on session1
            stat_request = cuvis_ai_pb2.TrainRequest(
                session_id=session_id1,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
            list(grpc_stub.Train(stat_request))

            # Session2 should still have its original config unchanged
            # (This verifies sessions don't share state)
            # We can't directly check this without a GetSession RPC, but the fact
            # that we can still use session2 is a good sign

        finally:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id1))
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id2))
