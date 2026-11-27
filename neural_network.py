from typing import cast
import numpy as np
import pickle
from layer_util import layer_sequence
from layers_affine import softmax_with_cee

class network:
    def __init__(self, sequence : layer_sequence) -> None:
        self.sequence = sequence

    def serialize(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def deserialize(filename: str) -> network:
        with open(filename, 'rb') as f:
            loaded_net = pickle.load(f)
        return cast(network, loaded_net)

    def forward(self, input : np.ndarray):
        return self.sequence.forward(input, None, False)

    def step_train(self, input : np.ndarray, label : np.ndarray, step_k):
        # 需要层中最后一项是 softmax_with_cee
        self.sequence.forward(input, label, True)
        last_layer = cast(softmax_with_cee, self.sequence.layers[-1])
        loss = last_layer.loss
        self.sequence.backward() # 输出层用 forward 时缓存的标签 t 计算梯度
        self.sequence.adjust((step_k,)) 
        return loss