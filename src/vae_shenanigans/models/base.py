import torch
from torch import nn
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, **kwargs) -> torch.Tensor:
        raise RuntimeError

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass
