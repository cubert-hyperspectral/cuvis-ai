"""Internal Lightning Module for training cuvis.ai graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
import torch
from loguru import logger

from cuvis_ai.pipeline.executor import MemoryExecutor

if TYPE_CHECKING:
    from cuvis_ai.pipeline.graph import Graph
    from cuvis_ai.training.config import TrainingConfig


class CuvisLightningModule(pl.LightningModule):
    """Internal Lightning module that wraps a cuvis.ai Graph for training.
    
    This module is not intended for direct instantiation by users. Instead, use
    `Graph.train()` which automatically creates and configures this module.
    
    The module orchestrates:
    - Forward passes through the graph
    - Loss aggregation from leaf nodes
    - Metric computation and logging
    - Visualization generation
    - Monitoring plugin communication
    
    Parameters
    ----------
    graph : Graph
        The cuvis.ai processing graph to train
    training_config : TrainingConfig
        Training configuration including optimizer, scheduler, and monitoring settings
    """

    def __init__(self, graph: Graph, training_config: TrainingConfig) -> None:
        super().__init__()
        self.graph = graph
        self.graph_modules = self.graph.torch_layers  # ← Add this line
        self.training_config = training_config

        # Save hyperparameters (excluding graph to avoid serialization issues)
        self.save_hyperparameters(ignore=["graph"])

        # Store config dict for reproducibility
        from cuvis_ai.training.config import as_dict
        self.hparams.update(as_dict(training_config))

    def forward(self, x: torch.Tensor, y=None, m=None) -> tuple:
        """Forward pass through the graph.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        y : Any, optional
            Target labels/masks
        m : dict, optional
            Additional metadata
            
        Returns
        -------
        tuple
            (x_out, y_out, m_out) from graph forward pass
        """
        return self.graph.forward(x, y, m)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step collecting losses from leaf nodes.
        
        Parameters
        ----------
        batch : dict
            Batch dictionary with keys "cube"/"x" (required) and "mask"/"labels" (optional)
        batch_idx : int
            Batch index
            
        Returns
        -------
        torch.Tensor
            Aggregated loss for backpropagation
        """
        # Extract data from batch
        x = batch.get("cube") if "cube" in batch else batch.get("x")
        y = batch.get("mask") if "mask" in batch else batch.get("labels")
        m = {k: v for k, v in batch.items() if k not in ["cube", "x", "mask", "labels"]}

        # Collect parent outputs for leaf nodes
        with torch.enable_grad():  # Ensure gradients are enabled
            parent_outputs = self._collect_parent_outputs(x, y, m)

        # Collect losses from loss leaf nodes
        losses = {}
        loss_infos = {}
        from cuvis_ai.training.leaf_nodes import LossNode

        for leaf_id, leaf_info in self.graph.leaf_nodes.items():
            if issubclass(leaf_info["family"], LossNode):
                loss_node = leaf_info["node"]
                parent_id = leaf_info["parent"]
                parent_out = parent_outputs[parent_id]

                # Compute loss (returns tuple: loss_value, info_dict)
                with torch.enable_grad():  # Ensure gradients flow through loss computation
                    loss_result = loss_node.compute_loss(
                        parent_output=parent_out,
                        labels=y,
                        metadata=m,
                    )

                # Handle tuple return (loss, info)
                if isinstance(loss_result, tuple):
                    loss_value, info = loss_result
                    loss_infos[leaf_id] = info
                else:
                    loss_value = loss_result

                losses[leaf_id] = loss_value

                # Log individual loss
                self.log(
                    f"train/loss/{leaf_id}",
                    loss_value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                )

                # Log additional info from loss computation
                if leaf_id in loss_infos:
                    for info_key, info_value in loss_infos[leaf_id].items():
                        self.log(
                            f"train/loss_info/{leaf_id}/{info_key}",
                            info_value,
                            on_step=True,
                            on_epoch=True,
                            prog_bar=False,
                        )

        # Aggregate losses (simple sum for now, could add weighting later)
        if not losses:
            logger.warning("No loss nodes found - training without losses!")
            return torch.tensor(0.0, requires_grad=True, device=x.device)

        total_loss = sum(losses.values())
        self.log("train/loss/total", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log to monitoring plugins
        if self.graph.monitoring_plugins:
            metrics_dict = {f"train/loss/{k}": v.item() for k, v in losses.items()}
            metrics_dict["train/loss/total"] = total_loss.item()

            # Add loss info to metrics
            for leaf_id, info in loss_infos.items():
                for info_key, info_value in info.items():
                    metrics_dict[f"train/loss_info/{leaf_id}/{info_key}"] = info_value

            for monitor in self.graph.monitoring_plugins:
                monitor.log_metrics(
                    metrics=metrics_dict,
                    step=self.global_step,
                    stage="train",
                )

        return total_loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Validation step computing metrics and generating visualizations.
        
        Parameters
        ----------
        batch : dict
            Batch dictionary with keys "cube"/"x" (required) and "mask"/"labels" (optional)
        batch_idx : int
            Batch index
        """
        # Extract data from batch
        x = batch.get("cube") if "cube" in batch else batch.get("x")
        y = batch.get("mask") if "mask" in batch else batch.get("labels")
        m = {k: v for k, v in batch.items() if k not in ["cube", "x", "mask", "labels"]}

        # Collect parent outputs for leaf nodes
        parent_outputs = self._collect_parent_outputs(x, y, m)

        # Compute validation losses (for monitoring)
        from cuvis_ai.training.leaf_nodes import LossNode
        val_losses = {}
        val_loss_infos = {}

        for leaf_id, leaf_info in self.graph.leaf_nodes.items():
            if issubclass(leaf_info["family"], LossNode):
                loss_node = leaf_info["node"]
                parent_id = leaf_info["parent"]
                parent_out = parent_outputs[parent_id]

                # Compute loss
                loss_result = loss_node.compute_loss(
                    parent_output=parent_out,
                    labels=y,
                    metadata=m,
                )

                if isinstance(loss_result, tuple):
                    loss_value, info = loss_result
                    val_loss_infos[leaf_id] = info
                else:
                    loss_value = loss_result

                val_losses[leaf_id] = loss_value

                # Log validation loss
                self.log(
                    f"val/loss/{leaf_id}",
                    loss_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        # Log total validation loss
        if val_losses:
            total_val_loss = sum(val_losses.values())
            self.log("val/loss/total", total_val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute metrics
        from cuvis_ai.training.leaf_nodes import MetricNode
        all_metrics = {}

        for leaf_id, leaf_info in self.graph.leaf_nodes.items():
            if issubclass(leaf_info["family"], MetricNode):
                metric_node = leaf_info["node"]
                parent_id = leaf_info["parent"]
                parent_out = parent_outputs[parent_id]

                # Compute metric (returns dict)
                metrics_dict = metric_node.compute_metric(
                    parent_output=parent_out,
                    labels=y,
                    metadata=m,
                )

                # Log each metric
                if isinstance(metrics_dict, dict):
                    for metric_name, metric_value in metrics_dict.items():
                        full_name = f"val/{metric_name}"
                        self.log(
                            full_name,
                            metric_value,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                        )
                        all_metrics[full_name] = metric_value

        # Log to monitoring plugins
        if self.graph.monitoring_plugins:
            metrics_dict = {}

            # Add validation losses
            for leaf_id, loss_value in val_losses.items():
                metrics_dict[f"val/loss/{leaf_id}"] = loss_value.item()
            if val_losses:
                metrics_dict["val/loss/total"] = sum(val_losses.values()).item()

            # Add loss info
            for leaf_id, info in val_loss_infos.items():
                for info_key, info_value in info.items():
                    metrics_dict[f"val/loss_info/{leaf_id}/{info_key}"] = info_value

            # Add metrics
            metrics_dict.update(all_metrics)

            for monitor in self.graph.monitoring_plugins:
                monitor.log_metrics(
                    metrics=metrics_dict,
                    step=self.global_step,
                    stage="val",
                )

        # Generate visualizations
        from cuvis_ai.training.leaf_nodes import VisualizationNode

        for leaf_id, leaf_info in self.graph.leaf_nodes.items():
            if issubclass(leaf_info["family"], VisualizationNode):
                viz_node = leaf_info["node"]
                parent_id = leaf_info["parent"]
                parent_out = parent_outputs[parent_id]

                # Generate visualization
                try:
                    # Extract parent output components
                    parent_tensor = parent_out[0]  # (x, y, m) tuple -> x
                    parent_y = parent_out[1] if len(parent_out) > 1 else y
                    parent_m = parent_out[2] if len(parent_out) > 2 else m

                    artifacts = viz_node.visualize(
                        parent_output=parent_tensor,
                        labels=parent_y,
                        metadata=parent_m,
                        stage="val",
                    )

                    # Log to monitoring plugins
                    if artifacts and self.graph.monitoring_plugins:
                        for monitor in self.graph.monitoring_plugins:
                            monitor.log_artifacts(
                                artifacts={f"{parent_id}_{viz_node.__class__.__name__}": artifacts},
                                stage="val",
                                step=self.global_step,
                            )
                except Exception as e:
                    logger.warning(f"Failed to generate visualization {leaf_id}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.
        
        Returns
        -------
        dict
            Dictionary containing optimizer and optionally lr_scheduler config
        """
        opt_config = self.training_config.optimizer

        # Get trainable parameters from graph
        trainable_params = list(self.graph.parameters(require_grad=True))

        if not trainable_params:
            logger.warning("No trainable parameters found in graph!")
            # Return dummy optimizer to avoid Lightning errors
            return {"optimizer": torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0)}

        # Create optimizer
        if opt_config.name.lower() == "adam":
            adam_kwargs = {
                "lr": opt_config.lr,
                "weight_decay": opt_config.weight_decay,
            }
            if opt_config.betas is not None:
                adam_kwargs["betas"] = opt_config.betas
            optimizer = torch.optim.Adam(trainable_params, **adam_kwargs)
        elif opt_config.name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=opt_config.lr,
                betas=opt_config.betas,
                weight_decay=opt_config.weight_decay,
            )
        elif opt_config.name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=opt_config.lr,
                momentum=opt_config.momentum,
                weight_decay=opt_config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config.name}")

        return {"optimizer": optimizer}

    def _collect_parent_outputs(
        self, x: torch.Tensor, y=None, m=None
    ) -> dict[str, tuple]:
        """Collect outputs from parent nodes in single forward pass.
        
        Maintains unified computational graph to avoid double backward errors.
        """
        import networkx as nx

        parent_outputs: dict[str, tuple[torch.Tensor, Any, Any]] = {}
        sorted_ids = list(nx.topological_sort(self.graph.graph))
        
        # Forward through entry node
        entry_id = self.graph.entry_point
        entry_node = self.graph.nodes[entry_id]
        
        out = entry_node.forward(x, y, m)
        if isinstance(out, tuple):
            curr_x, curr_y, curr_m = (out + (y, m))[:3] if len(out) < 3 else out
        else:
            curr_x, curr_y, curr_m = out, y, m
        
        parent_outputs[entry_id] = (curr_x, curr_y, curr_m)
        
        # Forward through remaining nodes in topological order
        for node_id in sorted_ids:
            if node_id == entry_id:
                continue
                
            node = self.graph.nodes[node_id]
            parent_ids = list(self.graph.graph.predecessors(node_id))
            
            if not parent_ids:
                continue
            
            # Gather inputs from parents
            if len(parent_ids) == 1:
                input_x, input_y, input_m = parent_outputs[parent_ids[0]]
            else:
                # Concatenate multiple parent outputs
                parent_x_list = [parent_outputs[p][0] for p in parent_ids]
                input_x = torch.cat(parent_x_list, dim=-1)
                
                # Handle labels (y)
                parent_y_list = [parent_outputs[p][1] for p in parent_ids if parent_outputs[p][1] is not None]
                if parent_y_list and all(torch.is_tensor(py) for py in parent_y_list):
                    input_y = torch.cat(parent_y_list, dim=-1)
                else:
                    input_y = parent_y_list[0] if parent_y_list else y
                
                # Use first parent's metadata
                input_m = parent_outputs[parent_ids[0]][2]
            
            # Forward through node
            out = node.forward(input_x, input_y, input_m)
            if isinstance(out, tuple):
                curr_x, curr_y, curr_m = (out + (input_y, input_m))[:3] if len(out) < 3 else out
            else:
                curr_x, curr_y, curr_m = out, input_y, input_m
            
            parent_outputs[node_id] = (curr_x, curr_y, curr_m)
        
        return parent_outputs

    def transfer_batch_to_device(self, batch: dict[str, Any], device: torch.device, dataloader_idx: int) -> dict[str, Any]:
        """Transfer custom dict batch structure to device.
        
        This handles both single-GPU and multi-GPU (DDP) scenarios.
        Lightning calls this automatically before training/validation steps.
        """
        batch_on_device = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch_on_device[key] = value.to(device, non_blocking=True)
            else:
                batch_on_device[key] = value
        return batch_on_device
    def setup(self, stage: str) -> None:
        """Setup hook called before training/validation/testing.
        
        This is called on every process in distributed training, and ensures
        the graph is moved to the correct device early in the setup process.
        """
        if stage == "fit":
            # Determine target device from trainer's strategy
            if hasattr(self.trainer, 'strategy') and hasattr(self.trainer.strategy, 'root_device'):
                device = self.trainer.strategy.root_device
                if device.type == "cuda":
                    self.graph.to(device)
                    logger.info(f"Moved graph to {device} during setup")
