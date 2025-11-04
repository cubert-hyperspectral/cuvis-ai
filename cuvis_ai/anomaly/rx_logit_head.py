"""RX Logit Head for Anomaly Detection

This module provides a trainable head that converts RX anomaly scores into
logits for binary anomaly classification. It can be trained end-to-end with
binary cross-entropy loss.
"""


import torch
import torch.nn as nn

from cuvis_ai.node import LabelLike, MetaLike, Node, NodeOutput


class RXLogitHead(Node):
    """Trainable head that converts RX scores to anomaly logits.
    
    This node takes RX anomaly scores (typically Mahalanobis distances) and
    applies a learned affine transformation to produce logits suitable for
    binary classification with BCEWithLogitsLoss.
    
    The transformation is: logit = scale * (score - bias)
    
    Parameters
    ----------
    init_scale : float, default=1.0
        Initial value for the scale parameter
    init_bias : float, default=0.0
        Initial value for the bias parameter (threshold)
    trainable : bool, default=True
        Whether scale and bias should be trainable parameters
    
    Attributes
    ----------
    scale : nn.Parameter or torch.Tensor
        Scale factor applied to scores
    bias : nn.Parameter or torch.Tensor
        Bias (threshold) subtracted from scores before scaling
    
    Examples
    --------
    >>> # After RX detector
    >>> rx = RXGlobal(eps=1e-6, trainable_stats=False)
    >>> logit_head = RXLogitHead(init_scale=1.0, init_bias=5.0, trainable=True)
    >>> graph.add_node(rx)
    >>> graph.add_node(logit_head, parent=rx)
    >>> 
    >>> # Add BCE loss for training
    >>> from cuvis_ai.training.losses import AnomalyBCEWithLogits
    >>> bce_loss = AnomalyBCEWithLogits(weight=1.0)
    >>> graph.add_leaf_node(bce_loss, parent=logit_head)
    """

    def __init__(
        self,
        init_scale: float = 1.0,
        init_bias: float = 0.0,
        trainable: bool = True,
    ):
        self.init_scale = init_scale
        self.init_bias = init_bias
        self.trainable = trainable

        super().__init__(
            init_scale=init_scale,
            init_bias=init_bias,
            trainable=trainable,
        )

        # Initialize parameters
        if trainable:
            self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
            self.bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))
        else:
            self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))
            self.register_buffer("bias", torch.tensor(init_bias, dtype=torch.float32))

    @property
    def input_dim(self) -> tuple[int, int, int, int]:
        """Expected input dimension: (B, H, W, 1) - RX scores."""
        return (-1, -1, -1, 1)

    @property
    def output_dim(self) -> tuple[int, int, int, int]:
        """Output dimension: (B, H, W, 1) - logits."""
        return (-1, -1, -1, 1)

    @property
    def requires_initial_fit(self) -> bool:
        """RXLogitHead can be initialized from statistics optionally."""
        return False

    @property
    def is_trainable(self) -> bool:
        """Whether this node has trainable parameters."""
        return self.trainable

    def initialize_from_data(self, iterator) -> None:
        """Initialize bias from statistics of RX scores.
        
        This is optional - the head can also be trained from scratch.
        If called, sets the bias to mean + 2*std of the scores.
        
        Parameters
        ----------
        iterator : Iterator
            Iterator yielding (scores, labels, metadata) tuples
        """
        if not self.trainable:
            return

        # Collect statistics from a sample of data
        score_list = []
        max_batches = 100  # Limit to avoid memory issues

        for i, (scores, _, _) in enumerate(iterator):
            if i >= max_batches:
                break
            if scores is not None:
                score_list.append(scores.detach().flatten())

        if not score_list:
            return

        # Compute statistics
        all_scores = torch.cat(score_list)
        mean = all_scores.mean()
        std = all_scores.std()

        # Set bias to mean + 2*std (common threshold for anomaly detection)
        with torch.no_grad():
            self.bias.copy_(mean + 2.0 * std)

    def forward(
        self,
        x: torch.Tensor,
        y: LabelLike = None,
        m: MetaLike = None,
    ) -> NodeOutput:
        """Transform RX scores to logits.
        
        Parameters
        ----------
        x : torch.Tensor
            Input RX scores with shape (B, H, W, 1)
        y : optional
            Labels/masks (passed through)
        m : optional
            Metadata (passed through)
        
        Returns
        -------
        tuple
            (logits, y, m) where logits have shape (B, H, W, 1)
        """
        # Ensure scale and bias are on the same device as x
        scale = self.scale.to(x.device, x.dtype)
        bias = self.bias.to(x.device, x.dtype)

        # Apply affine transformation: logit = scale * (score - bias)
        logits = scale * (x - bias)

        return logits, y, m

    def serialize(self, serial_dir: str) -> dict:
        """Serialize RXLogitHead state."""
        return {
            "params": {
                "init_scale": self.init_scale,
                "init_bias": self.init_bias,
                "trainable": self.trainable,
            },
            "state_dict": self.state_dict(),
        }

    def load(self, params: dict, serial_dir: str) -> None:
        """Load RXLogitHead state from serialized data."""
        config = params.get("params", {})
        self.init_scale = config.get("init_scale", self.init_scale)
        self.init_bias = config.get("init_bias", self.init_bias)
        self.trainable = config.get("trainable", self.trainable)

        state = params.get("state_dict", {})
        if state:
            self.load_state_dict(state, strict=False)

    def get_threshold(self) -> float:
        """Get the current anomaly threshold (bias value).
        
        Returns
        -------
        float
            Current threshold value
        """
        return self.bias.item()

    def set_threshold(self, threshold: float) -> None:
        """Set the anomaly threshold (bias value).
        
        Parameters
        ----------
        threshold : float
            New threshold value
        """
        with torch.no_grad():
            self.bias.fill_(threshold)

    def predict_anomalies(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to binary anomaly predictions.
        
        Parameters
        ----------
        logits : torch.Tensor
            Logits from forward pass, shape (B, H, W, 1)
        
        Returns
        -------
        torch.Tensor
            Binary predictions (0=normal, 1=anomaly), shape (B, H, W, 1)
        """
        return (logits > 0).float()
