from typing import Tuple
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from layer_util import layer

class reshape(layer):
    """把输入 reshape 成 (N, *shape)，用于把 (N, 784) 的 MNIST 数据送进卷积层。

    用法：reshape(1, 28, 28)，只传目标形状，不要传输入形状。
    """

    def __init__(self, *shape: int):
        self.shape = shape

    def forward(self, x: np.ndarray, use_cache: bool = True) -> np.ndarray:
        if use_cache:
            self.input_shape = x.shape
        return x.reshape(x.shape[0], *self.shape)

    def backward(self, dl_dy: np.ndarray) -> Tuple:
        return (dl_dy.reshape(self.input_shape),)

    def adjust(self, grad: Tuple, args: Tuple):
        pass # 无需更新


class flatten(layer):
    """把 (N, C, H, W) 的卷积/池化输出展平成 (N, C*H*W)，以接入 affine 层。

    前向只改变形状，不改变数值；反向时按缓存的原形状还原梯度。
    """

    def forward(self, x: np.ndarray, use_cache: bool = True) -> np.ndarray:
        if use_cache:
            self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dl_dy: np.ndarray) -> Tuple:
        return (dl_dy.reshape(self.input_shape),)

    def adjust(self, grad: Tuple, args: Tuple):
        pass # 无需更新


class convolution(layer):
    def __init__(self, N : int, C : int, Hx : int, Wx : int, Hf : int, Wf : int, P : int, S : int) -> None:
        # 与 affine 一致的 He 初始化；bias 从 0 开始。
        # 若用 N(0,1) 初始化，第二层卷积的激活会非常大，softmax 直接饱和、loss 卡在 ln(10)。
        fan_in = C * Hf * Wf
        self.filter = np.random.standard_normal((N, C, Hf, Wf)) * np.sqrt(2.0 / fan_in)
        self.bias = np.zeros(N)
        self.n, self.c, self.p, self.s = N, C, P, S
        self.hf, self.wf = Hf, Wf
        self.hx_pad, self.wx_pad = Hx + 2 * P, Wx + 2 * P
        self.hy = (self.hx_pad - Hf) // S + 1
        self.wy = (self.wx_pad - Wf) // S + 1

    @staticmethod
    def im2col_input(X, H_F, W_F, H_Y, W_Y, stride=1, padding=0):
        R, C, _, _ = X.shape
        if padding > 0:
            X = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        windows = sliding_window_view(X, window_shape=(H_F, W_F), axis=(2, 3))
        windows = windows[:, :, ::stride, ::stride, :, :]
        windows = windows.transpose(0, 2, 3, 1, 4, 5)
        return np.ascontiguousarray(windows).reshape(R * H_Y * W_Y, C * H_F * W_F)

    def forward(self, X: np.ndarray, use_cache: bool = True) -> np.ndarray:
        R = X.shape[0]
        X_prime = self.im2col_input(X, self.hf, self.wf, self.hy, self.wy, self.s, self.p)
        if use_cache:
            self.X_p_cache = X_prime
        F_prime = self.filter.transpose(1, 2, 3, 0).reshape(self.c * self.hf * self.wf, self.n)
        Y_prime = X_prime @ F_prime
        y = Y_prime.reshape(R, self.hy, self.wy, self.n).transpose(0, 3, 1, 2)
        y += self.bias.reshape(1, self.n, 1, 1) 
        return y

    def backward(self, dl_dy: np.ndarray) -> Tuple:
        R = dl_dy.shape[0]
        dl_dy_prime = dl_dy.transpose(0, 2, 3, 1).reshape(R * self.hy * self.wy, self.n)
        
        f_prime = self.filter.transpose(1, 2, 3, 0).reshape(-1, self.n)
        dl_dx_prime = dl_dy_prime @ f_prime.T
        dl_df_prime = self.X_p_cache.T @ dl_dy_prime
        
        dl_df = dl_df_prime.reshape(self.c, self.hf, self.wf, self.n).transpose(3, 0, 1, 2)
        dl_db = np.sum(dl_dy_prime, axis=0) 
        
        # col2im 还原 dl_dx：按卷积核内的偏移量 (kh, kw) 批量累加，
        # 避免对 hy*wy 个输出位置做 Python 双重循环（784 次 -> 25 次）
        dl_dx_padded = np.zeros((R, self.c, self.hx_pad, self.wx_pad), dtype=dl_dx_prime.dtype)
        dl_dx_prime_reshaped = dl_dx_prime.reshape(R, self.hy, self.wy, self.c, self.hf, self.wf)
        
        for kh in range(self.hf):
            for kw in range(self.wf):
                src = dl_dx_prime_reshaped[:, :, :, :, kh, kw].transpose(0, 3, 1, 2)
                dl_dx_padded[:, :, kh:kh + self.hy * self.s:self.s,
                             kw:kw + self.wy * self.s:self.s] += src
                
        if self.p > 0:
            dl_dx = dl_dx_padded[:, :, self.p:-self.p, self.p:-self.p]
        else:
            dl_dx = dl_dx_padded
            
        return (dl_dx, dl_df, dl_db)

    def adjust(self, grad: Tuple, args: Tuple):
        # 从 backward 的返回值中提取权重梯度
        _, dl_df, dl_db = grad
        lr = args[0] # 取学习率
        self.filter -= lr * dl_df
        self.bias -= lr * dl_db


class pooling_max(layer):
    def __init__(self, xd: int, yd: int, s: int) -> None:
        self.xd, self.yd, self.s = xd, yd, s

    @staticmethod
    def im2col_pool(X, xd, yd, stride=1):
        R, C, H_X, W_X = X.shape
        H_Y = (H_X - xd) // stride + 1
        W_Y = (W_X - yd) // stride + 1
        
        windows = sliding_window_view(X, window_shape=(xd, yd), axis=(2, 3))
        windows = windows[:, :, ::stride, ::stride, :, :]
        X_prime = np.ascontiguousarray(windows).reshape(R * C * H_Y * W_Y, xd * yd)
        
        window_offset = np.arange(xd * yd)
        rel_h, rel_w = window_offset // yd, window_offset % yd
        h_grid, w_grid = np.meshgrid(np.arange(H_Y) * stride, np.arange(W_Y) * stride, indexing='ij')
        
        abs_h = h_grid.ravel()[:, np.newaxis] + rel_h
        abs_w = w_grid.ravel()[:, np.newaxis] + rel_w
        abs_indices = np.tile((abs_h * W_X + abs_w), (R * C, 1))
        
        return X_prime, abs_indices, H_Y, W_Y

    def forward(self, X: np.ndarray, use_cache: bool = True) -> np.ndarray:
        self.X_shape = X.shape
        x_prime, abs_indices, hy, wy = self.im2col_pool(X, self.xd, self.yd, self.s)
        
        if use_cache:
            rel_indices = np.argmax(x_prime, axis=1)
            self.max_abs_indices = abs_indices[np.arange(len(rel_indices)), rel_indices]
        
        y_prime = np.max(x_prime, axis=1)
        self.output_shape = (X.shape[0], X.shape[1], hy, wy)
        return y_prime.reshape(self.output_shape)

    def backward(self, dl_dy: np.ndarray) -> Tuple:
        R, C, _, _ = self.X_shape
        dX = np.zeros(self.X_shape, dtype=dl_dy.dtype)
        
        dY_flat = dl_dy.reshape(R * C, -1)
        dX_flat = dX.reshape(R * C, -1)
        idx_flat = self.max_abs_indices.reshape(R * C, -1)
        
        batch_idx = np.arange(R * C)[:, np.newaxis]
        np.add.at(dX_flat, (batch_idx, idx_flat), dY_flat)
            
        return (dX,)

    def adjust(self, grad: Tuple, args: Tuple):
        pass # 无需更新