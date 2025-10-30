from __future__ import annotations

from pathlib import Path

import torch

from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.anomoly.rx_v2 import RXPerBatch
from cuvis_ai.data.lentils_anomoly import LentilsAnomoly
from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.pipeline.graph import Graph


def _prepare_datamodule(data_root: Path, batch_size: int = 1) -> LentilsAnomoly:
    """Download (if needed) and load the Lentils anomaly data module."""
    datamodule = LentilsAnomoly(str(data_root), batch_size=batch_size)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    return datamodule


def _ensure_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Force a tensor into BHWC layout with an explicit batch dimension."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"Expected BHWC tensor, received shape {tuple(x.shape)}")
    return x



def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path("D:\\code-repos\\cuvis.ai\\data\\Lentils")
    datamodule = _prepare_datamodule(data_root, batch_size=1)

    graph = Graph("rx_anomaly_demo")
    rx_node = RXPerBatch()
    normalizer = MinMaxNormalizer()
    decider = BinaryDecider(threshold=0.5)

    graph.add_node(rx_node)
    graph.add_node(normalizer, parent=rx_node)
    graph.add_node(decider, parent=normalizer)

    # Move modules to the selected device.
    graph.to(device)

    batch = next(iter(datamodule.val_dataloader()))
    cubes_array = batch["cube"]
    if not isinstance(cubes_array, torch.Tensor):
        cubes_tensor = torch.from_numpy(cubes_array)
    else:
        cubes_tensor = cubes_array
    cubes = _ensure_bhwc(cubes_tensor).to(device)
    decisions, _, _ = graph.forward(cubes)
    decisions = (decisions.detach() >= 0.5).to(torch.int32)
    
    # with torch.no_grad():
    #     raw_scores = rx_node.rx(cubes).detach()
    #     normalized_scores = normalizer(raw_scores.unsqueeze(-1))[0].detach().squeeze(-1)

    
    # print(
    #     f"RXPerBatch raw scores -> shape: {tuple(raw_scores.shape)}, "
    #     f"min: {raw_scores.min().item():.4f}, max: {raw_scores.max().item():.4f}"
    # )
    # print(
    #     f"MinMax normalized -> shape: {tuple(normalized_scores.shape)}, "
    #     f"min: {normalized_scores.min().item():.4f}, max: {normalized_scores.max().item():.4f}"
    # )


    print(
        "Binary decisions -> "
        f"total anomalous pixels: {int(decisions.sum().item())} of {decisions.numel()}"
    )


if __name__ == "__main__":
    main()
