import numpy as np
from abc import ABC, abstractmethod
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional, Tuple

class layer(ABC):
    def __init__(self):
        self.training = True

    @abstractmethod
    def forward(self, x: np.ndarray, use_cache: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dl_dy: np.ndarray | None) -> Tuple:
        """
        反向传播
        返回: (dl_dx, ...)（第一个参数是传给上游用的梯度，其余梯度用于更新自身
        """
        pass

    @abstractmethod
    def adjust(self, grad : Tuple, args : Tuple):
        """
        根据梯度调整网络权重参数
        grad 的第一项是传给上游用的梯度，其余梯度是有用的
        args 中可以包含 step_k 等超参数
        """
        pass

class loss_layer(layer):
    @abstractmethod
    def forward(self, x: np.ndarray, t: np.ndarray | None = None, use_cache: bool = True) -> np.ndarray:
        pass

class layer_sequence(layer):
    def __init__(self, *layers: layer):
        super().__init__()
        self.layers = list(layers)
        self._grad_cache = [] 

    def forward(self, x: np.ndarray, t: np.ndarray | None = None, use_cache: bool = True) -> np.ndarray:
        for l in self.layers:
            if isinstance(l, loss_layer):
                x = l.forward(x, t, use_cache)
            else:
                x = l.forward(x, use_cache)
        return x

    def backward(self, dl_dy: np.ndarray | None = None) -> Tuple:
        self._grad_cache.clear()
        for l in reversed(self.layers):
            grads = l.backward(dl_dy)
            self._grad_cache.append((l, grads))
            dl_dy = grads[0]
        return (dl_dy,)

    def adjust(self, args: Tuple):
        for l, grads in self._grad_cache:
            l.adjust(grads, args)
        self._grad_cache.clear()