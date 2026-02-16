from __future__ import annotations

import numpy as np
import torch

from cuvis_ai.node.data import CU3SDataNode, LentilsAnomalyDataNode


def test_cu3s_data_node_converts_cube_and_passthroughs_mask_and_wavelengths(create_test_cube):
    node = CU3SDataNode()

    cube, wavelengths = create_test_cube(
        batch_size=2,
        height=4,
        width=5,
        num_channels=6,
        mode="random",
        dtype=torch.uint16,
    )
    mask = torch.randint(
        low=0,
        high=4,
        size=(2, 4, 5),
        dtype=torch.int32,
    )

    out = node.forward(cube=cube, mask=mask, wavelengths=wavelengths)

    assert out["cube"].dtype == torch.float32
    assert out["cube"].shape == cube.shape
    assert torch.equal(out["cube"], cube.to(torch.float32))

    assert "mask" in out
    assert out["mask"].dtype == torch.int32
    assert torch.equal(out["mask"], mask)

    assert "wavelengths" in out
    assert isinstance(out["wavelengths"], np.ndarray)
    assert out["wavelengths"].dtype == np.int32
    np.testing.assert_array_equal(out["wavelengths"], wavelengths[0].cpu().numpy())


def test_cu3s_data_node_handles_optional_mask_and_wavelengths(create_test_cube):
    node = CU3SDataNode()

    cube, _ = create_test_cube(
        batch_size=1,
        height=2,
        width=3,
        num_channels=4,
        mode="random",
        dtype=torch.uint16,
    )

    out = node.forward(cube=cube, mask=None, wavelengths=None)

    assert set(out.keys()) == {"cube"}
    assert out["cube"].dtype == torch.float32
    assert out["cube"].shape == cube.shape


def test_lentils_data_node_inherits_cu3s_data_node():
    assert issubclass(LentilsAnomalyDataNode, CU3SDataNode)


def test_lentils_data_node_uses_cu3s_normalization_and_binary_mask_mapping(create_test_cube):
    node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])

    cube, wavelengths = create_test_cube(
        batch_size=1,
        height=2,
        width=3,
        num_channels=4,
        mode="random",
        dtype=torch.uint16,
    )
    mask = torch.tensor(
        [[[0, 1, 2], [3, 0, 1]]],
        dtype=torch.int32,
    )

    out = node.forward(cube=cube, mask=mask, wavelengths=wavelengths)

    # Reused normalization behavior from CU3SDataNode.
    assert out["cube"].dtype == torch.float32
    assert isinstance(out["wavelengths"], np.ndarray)
    assert out["wavelengths"].dtype == np.int32
    np.testing.assert_array_equal(out["wavelengths"], wavelengths[0].cpu().numpy())

    # Lentils-specific binary mapping behavior.
    assert out["mask"].dtype == torch.bool
    assert out["mask"].shape == (1, 2, 3, 1)
    expected = torch.tensor(
        [[[[False], [False], [True]], [[True], [False], [False]]]],
        dtype=torch.bool,
    )
    assert torch.equal(out["mask"], expected)
