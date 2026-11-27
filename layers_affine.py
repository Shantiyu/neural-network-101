from typing import Tuple
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from layer_util import layer, loss_layer

class relu(layer):
    def forward(self, x: np.ndarray, use_cache: bool = True) -> np.ndarray:
        if use_cache:
            # 缓存 mask 供反向传播使用
            self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, dl_dy: np.ndarray) -> Tuple:
        # 无参数梯度
        return (dl_dy * self.mask,)

    def adjust(self, grad: Tuple, args: Tuple):
        pass # 无需更新

class softmax_with_cee(loss_layer):
    def forward(self, x: np.ndarray, t: np.ndarray | None = None, use_cache: bool = True) -> np.ndarray:
        c = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - c)
        self.probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        self.t = t
        if t is not None:
            self.loss = -np.sum(t * np.log(self.probs + 1e-8)) / len(x)
        return self.probs

    def backward(self, dl_dy: np.ndarray | None = None) -> Tuple:
        # loss 是对 batch 取均值，所以梯度也要除以 batch size
        if self.t is None:
            raise RuntimeError("softmax_with_cee.backward() 需要先带标签 t 执行 forward()")
        dl_dx = (self.probs - self.t) / len(self.probs)
        return (dl_dx,)

    def adjust(self, grad: Tuple, args: Tuple):
        pass

class affine(layer):
    def __init__(self, in_features: int, out_features: int) -> None:
        self.weight = np.random.standard_normal((in_features, out_features)) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros(out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: np.ndarray, use_cache: bool = True) -> np.ndarray:
        # Z = XW + B
        if use_cache:
            self.x_cache = x
        return x @ self.weight + self.bias

    def backward(self, dl_dy: np.ndarray) -> Tuple:
        dl_dx = dl_dy @ self.weight.T
        dl_dw = self.x_cache.T @ dl_dy
        dl_db = np.sum(dl_dy, axis=0)
        return (dl_dx, dl_dw, dl_db)

    def adjust(self, grad: Tuple, args: Tuple):
        _, dl_dw, dl_db = grad
        lr = args[0]
        self.weight -= lr * dl_dw
        self.bias -= lr * dl_db