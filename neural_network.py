import numpy as np
import pickle
from backward import *

# 这个代码不允许开 copilot，你他妈再不练习手写代码就要不会写代码了

class network:
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('z_cache', None)
        state.pop('q_cache', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(self, layer_size : list):
        self.weight : list[np.ndarray] = []
        self.shift : list[np.ndarray] = []
        for i in range(len(layer_size)):
            if i >= 1:
                self.weight.append(np.zeros((layer_size[i - 1], layer_size[i]), np.float64))
                self.shift.append(np.zeros((layer_size[i])))

    # 序列化
    def serialize(self, path : str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def deserialize(path : str):
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        return loaded

    # ReLU 激活函数
    @staticmethod
    def relu(x : np.ndarray) -> np.ndarray:
        mask = (x <= 0)
        out = x
        out[mask] = 0
        return out

    # sigmoid 激活函数
    @staticmethod
    def sigmoid(x : np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    # softmax 输出层函数
    @staticmethod
    def softmax(x : np.ndarray) -> np.ndarray:
        c = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - c)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    

    def random_init(self):
        # randn 会以 x = 0, σ = 1 的参数生成正态分布。可以提供维数。
        for i, w in enumerate(self.weight):
            self.weight[i] = np.random.standard_normal(w.shape) * np.sqrt(2.0 / w.shape[0])

    def print_weight(self):
        print("weight:")
        for w in self.weight:
            print(w)
        print("shift:")
        for b in self.shift:
            print(b)

    # 向前传播一层
    def step_forward(self, previous : np.ndarray, layer_index : int, use_relu = True) -> np.ndarray:
        # Y = f(XW + B)，这里 XW 是矩阵乘法
        result = previous @ self.weight[layer_index] + self.shift[layer_index]
        if use_relu:
            result = self.relu(result)
        return result

    def step_affine_backward(self, dl_dz_l : np.ndarray, w_l : np.ndarray, z_lm1 : np.ndarray, z_l : np.ndarray, use_relu = True):
        dl_dy_l = relu_dl_dy(dl_dz_l, z_l) if use_relu else dl_dz_l # y_l > 0 等价于 z_l > 0，所以可以直接等价着用！
        dl_dz_lm1 = affine_dl_dz(w_l, dl_dy_l)
        dl_dw_l = affine_dl_dw(z_lm1, dl_dy_l)
        dl_db_l = affine_dl_db(dl_dy_l)
        return (dl_dz_lm1, dl_dw_l, dl_db_l)

    # 前向传播
    def forward(self, input : np.ndarray, use_cache = False) -> np.ndarray:
        vector = input
        if use_cache:
            self.z_cache = [vector]
        l = len(self.weight)
        for i in range(l):
            vector = self.step_forward(vector, i, i != l - 1)
            if use_cache:
                self.z_cache.append(vector)
        vector = self.softmax(vector)
        self.q_cache = vector
        return vector

    def backward(self, t : np.ndarray):
        N = len(self.weight) + 1
        dl_dz = [np.array([])] * N
        dl_dw = [np.array([])] * N
        dl_db = [np.array([])] * N
        N -= 1
        dl_dz[N] = softmax_cee_dl_dz(self.q_cache, t)
        for i in range(N, 0, -1):
            dl_dz[i - 1], dl_dw[i], dl_db[i] = self.step_affine_backward(dl_dz[i], self.weight[i - 1], self.z_cache[i - 1], self.z_cache[i], i != N)
        return (dl_dz, dl_dw, dl_db)

    def adjust(self, step_k : float, dl_dw : list[np.ndarray], dl_db : list[np.ndarray]) -> None:
        N = len(self.weight)
        for i in range(N):
            self.weight[i] -= step_k * dl_dw[i + 1]
            self.shift[i] -= step_k * dl_db[i + 1]


# test
if __name__ == '__main__':
    net = network([5, 10, 10, 2])
    net.random_init()
    #net.print_weight()
    data = net.forward(np.array([1, 2, 3, 4, 5]))
    print(data)