from abc import ABC, abstractmethod
from dataclasses import dataclass


class Optimizer(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def args(self):
        pass

    @property
    @abstractmethod
    def sklearn_args(self):
        pass

    @property
    @abstractmethod
    def pytorch_args(self):
        pass


@dataclass
class Adam(Optimizer):
    lr: float = 0.001
    alpha: float = 0.0001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8

    @Optimizer.name.getter
    def name(self):
        return "Adam Optimizer"

    @Optimizer.args.getter
    def args(self):
        return {
            "learning_rate_init": self.lr,
            "alpha": self.alpha,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
        }

    @Optimizer.sklearn_args.getter
    def sklearn_args(self):
        return {
            "learning_rate_init": self.lr,
            "alpha": self.alpha,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
        }

    @Optimizer.pytorch_args.getter
    def pytorch_args(self):
        try:
            import torch.optim
        except ImportError as exc:
            msg = "This feature requires pytorch to be installed"
            raise ImportError(msg) from exc

        return {
            "cls": torch.optim.Adam,
            "lr": self.lr,
            "weight_decay": self.alpha,
            "betas": (self.beta_1, self.beta_2),
            "eps": self.epsilon,
        }


@dataclass
class SGD(Optimizer):
    lr: float = 0.001
    alpha: float = 0.0001
    power_t: float = 0.5
    momentum: float = 0.9
    nesterov: bool = True

    @Optimizer.name.getter
    def name(self):
        return "SGD Optimizer"

    @Optimizer.args.getter
    def args(self):
        return {
            "learning_rate_init": self.lr,
            "alpha": self.alpha,
            "power_t": self.power_t,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
        }

    @Optimizer.sklearn_args.getter
    def sklearn_args(self):
        return {
            "learning_rate_init": self.lr,
            "alpha": self.alpha,
            "power_t": self.power_t,
            "momentum": self.momentum,
            "nesterov_momentum": self.nesterov,
        }

    @Optimizer.pytorch_args.getter
    def pytorch_args(self):
        try:
            import torch.optim
        except ImportError as exc:
            msg = "This feature requires pytorch to be installed"
            raise ImportError(msg) from exc

        return {
            "cls": torch.optim.SGD,
            "lr": self.lr,
            "weight_decay": self.alpha,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "dampening": self.power_t,
        }
