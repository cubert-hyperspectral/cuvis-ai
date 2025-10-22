from dataclasses import dataclass, field

import numpy as np
from skorch import NeuralNetClassifier
from torch import nn
from torch.optim.optimizer import Optimizer as torch_optim

from cuvis_ai.node.base import BaseSupervised
from cuvis_ai.utils.nn_config import Optimizer
from cuvis_ai.utils.numpy import flatten_labels, flatten_spatial, unflatten_spatial


@dataclass
class SkorchSupervised(BaseSupervised):
    epochs: int = 10
    optimizer: Optimizer | torch_optim = None
    verbose: bool = False
    model: nn.Module = None
    model_args: dict = field(default_factory=dict)

    def __post_init__(self):
        self.initialized = False

        args = dict()
        # if self.optimizer is not None:
        #    args.update(self.optimizer.args)

        model_args = {f"module__{k}": v for k, v in self.model_args.items()}

        self.classifier = NeuralNetClassifier(self.model, **args, **model_args)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        flatten_image = flatten_spatial(X)

        flatten_l = flatten_labels(Y)

        print(f"shape image: {flatten_image.shape}")
        print(f"shape labels: {flatten_l.shape}")

        self.classifier.fit(flatten_image, flatten_l)

    def forward(self, X: np.ndarray):
        flatten_image = flatten_spatial(X)

        predictions = self.classifier.predict(flatten_image)
        predictions = unflatten_spatial(predictions, X.shape)
        return predictions

    def serialize(self):
        pass

    def load():
        pass
