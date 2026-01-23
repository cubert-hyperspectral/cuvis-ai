import numpy as np
import torch
import torch.nn as nn

from cuvis_ai.anomaly.lad_detector import LADGlobal


def _lad_fit_numpy(arr_bhwc: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Reference NumPy implementation of LAD fit (no preprocessing)."""
    x = arr_bhwc[0].detach().cpu().numpy()  # H,W,C
    H, W, C = x.shape
    X = x.reshape(-1, C)
    M = X.mean(axis=0)
    A = np.abs(M[:, None] - M[None, :])
    a = float(M.mean())
    A = 1.0 / (1.0 + (A / (a + 1e-12)) ** 2)
    np.fill_diagonal(A, 0.0)
    D = np.diag(A.sum(axis=1))
    L = D - A
    d = np.diag(D)
    d_inv_sqrt = np.where(d > 0, 1.0 / (np.sqrt(d) + 1e-12), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = D_inv_sqrt @ L @ D_inv_sqrt
    return M.astype(np.float64), L.astype(np.float64)


def _lad_score_numpy(arr_bhwc: torch.Tensor, M: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Reference NumPy implementation of LAD scoring."""
    x = arr_bhwc[0].detach().cpu().numpy()
    H, W, C = x.shape
    X = x.reshape(-1, C)
    Xm = X - M
    out = np.einsum("ij,jk,ik->i", Xm, L, Xm)
    return out.reshape(H, W)


def _synthetic_cube(H: int, W: int, C: int, seed: int = 0) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, size=(H, W, C)).astype(np.float32)
    # add mild trend across channels
    trend = np.linspace(-0.2, 0.2, C, dtype=np.float32)
    base += trend.reshape(1, 1, C)
    x = torch.from_numpy(base).unsqueeze(0)  # [1,H,W,C]
    waves = np.linspace(430.0, 910.0, C, dtype=np.float64)
    return x, waves


def test_lad_m_l_numpy_parity():
    """Check that LADGlobal mean and Laplacian match NumPy reference."""
    H, W, C = 32, 32, 40
    x, _ = _synthetic_cube(H, W, C, seed=123)

    lad = LADGlobal(num_channels=C, use_numpy_laplacian=True)

    def stream():
        yield {"data": x}

    lad.statistical_initialization(stream())

    # NumPy reference
    M_np, L_np = _lad_fit_numpy(x)

    M_th = lad.M.detach().cpu().numpy()
    L_th = lad.L.detach().cpu().numpy()

    assert np.allclose(M_th, M_np, atol=1e-6, rtol=1e-6)
    assert np.allclose(L_th, L_np, atol=1e-4, rtol=1e-4)


def test_lad_scores_numpy_parity():
    """Check that LADGlobal scores match NumPy reference on separate train/test cubes."""
    H, W, C = 32, 32, 40
    x_train, _ = _synthetic_cube(H, W, C, seed=1)
    x_test, _ = _synthetic_cube(H, W, C, seed=2)
    # inject an anomaly patch in test
    x_test[:, 8:12, 8:12, -10:] += 0.8

    lad = LADGlobal(num_channels=C, use_numpy_laplacian=True)

    def train_stream():
        yield {"data": x_train}

    lad.statistical_initialization(train_stream())

    with torch.no_grad():
        out = lad.forward(x_test)
        s_th = out["scores"]

    s_th_np = s_th.squeeze(0).squeeze(-1).cpu().numpy()

    # NumPy reference
    M_np, L_np = _lad_fit_numpy(x_train)
    s_np = _lad_score_numpy(x_test, M_np, L_np)

    assert np.allclose(s_th_np, s_np, atol=1e-4, rtol=1e-4)


def test_lad_serialize_roundtrip(tmp_path):
    """Check that serialize/load round-trip preserves scores."""
    H, W, C = 16, 16, 20
    x, _ = _synthetic_cube(H, W, C, seed=7)

    lad = LADGlobal(num_channels=C, use_numpy_laplacian=True)

    def stream():
        yield {"data": x}

    lad.statistical_initialization(stream())

    state = lad.state_dict()

    lad2 = LADGlobal(num_channels=C)
    lad2.load_state_dict(state)

    with torch.no_grad():
        s1 = lad.forward(x)["scores"]
        s2 = lad2.forward(x)["scores"]

    assert torch.allclose(s1, s2, atol=1e-7, rtol=1e-7)


def test_lad_serialize_roundtrip_after_unfreeze(tmp_path):
    """Check that serialize/load round-trip preserves scores even after unfreeze."""
    H, W, C = 16, 16, 20
    x, _ = _synthetic_cube(H, W, C, seed=7)

    lad = LADGlobal(num_channels=C, use_numpy_laplacian=True)

    def stream():
        yield {"data": x}

    lad.statistical_initialization(stream())
    lad.unfreeze()  # Convert to parameters

    # Verify they are parameters
    assert isinstance(lad.M, nn.Parameter)
    assert isinstance(lad.L, nn.Parameter)

    state = lad.state_dict()

    lad2 = LADGlobal(num_channels=C)
    lad2.load_state_dict(state)

    # After loading, M and L should be restored (as buffers or parameters depending on state_dict)
    # The important thing is that scores match
    with torch.no_grad():
        s1 = lad.forward(x)["scores"]
        s2 = lad2.forward(x)["scores"]

    assert torch.allclose(s1, s2, atol=1e-7, rtol=1e-7)


def test_lad_unfreeze_converts_buffers_to_parameters():
    """Test that unfreeze() converts M and L buffers to trainable parameters."""
    H, W, C = 16, 16, 20
    x, _ = _synthetic_cube(H, W, C, seed=7)

    lad = LADGlobal(num_channels=C, use_numpy_laplacian=True)

    def stream():
        yield {"data": x}

    lad.statistical_initialization(stream())

    # Initially M and L should be buffers
    assert isinstance(lad.M, torch.Tensor)
    assert isinstance(lad.L, torch.Tensor)
    assert not isinstance(lad.M, nn.Parameter)
    assert not isinstance(lad.L, nn.Parameter)
    assert not lad.M.requires_grad
    assert not lad.L.requires_grad

    # Store original values
    M_original = lad.M.clone()
    L_original = lad.L.clone()

    # Unfreeze
    lad.unfreeze()

    # After unfreeze, M and L should be parameters with requires_grad=True
    assert isinstance(lad.M, nn.Parameter)
    assert isinstance(lad.L, nn.Parameter)
    assert lad.M.requires_grad
    assert lad.L.requires_grad

    # Values should be preserved
    assert torch.allclose(lad.M, M_original, atol=1e-7)
    assert torch.allclose(lad.L, L_original, atol=1e-7)

    # Node should not be frozen
    assert not lad.freezed


def test_lad_trainable_parameters_update(synthetic_anomaly_datamodule, training_config_factory):
    """Test that LAD parameters can be updated during gradient training."""
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training.trainers import GradientTrainer, StatisticalTrainer

    from cuvis_ai.node.data import LentilsAnomalyDataNode
    from cuvis_ai.node.losses import AnomalyBCEWithLogits
    from cuvis_ai.node.normalization import MinMaxNormalizer

    # Create datamodule using fixture (includes wavelengths automatically)
    datamodule = synthetic_anomaly_datamodule(
        batch_size=2,
        num_samples=16,
        height=8,
        width=8,
        channels=20,
        seed=42,
        include_labels=True,
        mode="random",
    )

    # Build pipeline
    pipeline = CuvisPipeline("test_lad_training")
    data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
    normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    lad = LADGlobal(num_channels=20, eps=1.0e-8, normalize_laplacian=True, use_numpy_laplacian=True)
    loss_node = AnomalyBCEWithLogits(weight=1.0)

    pipeline.connect(
        (data_node.outputs.cube, normalizer.data),
        (normalizer.normalized, lad.data),
        (lad.scores, loss_node.predictions),
        (data_node.outputs.mask, loss_node.targets),
    )

    # Statistical initialization
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # Store initial parameter values
    assert isinstance(lad.M, torch.Tensor)
    assert isinstance(lad.L, torch.Tensor)
    M_initial = lad.M.clone()
    L_initial = lad.L.clone()

    # Unfreeze to make trainable
    lad.unfreeze()

    # Verify parameters are trainable
    assert isinstance(lad.M, nn.Parameter)
    assert isinstance(lad.L, nn.Parameter)
    assert lad.M.requires_grad
    assert lad.L.requires_grad

    # Gradient training using config factory
    training_cfg = training_config_factory(max_epochs=2, lr=1e-2)

    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[loss_node],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
    )
    grad_trainer.fit()

    # Verify parameters were updated
    M_final = lad.M.data.clone()
    L_final = lad.L.data.clone()

    M_change = torch.norm(M_final - M_initial).item()
    L_change = torch.norm(L_final - L_initial).item()

    # Parameters should have changed during training
    assert M_change > 1e-6 or L_change > 1e-6, (
        f"LAD parameters did not change during training (M_change={M_change:.6e}, L_change={L_change:.6e})"
    )

    # Verify parameters are still trainable
    assert isinstance(lad.M, nn.Parameter)
    assert isinstance(lad.L, nn.Parameter)
    assert lad.M.requires_grad
    assert lad.L.requires_grad
