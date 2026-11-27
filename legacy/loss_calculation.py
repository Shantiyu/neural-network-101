from neural_network import network
import numpy as np

def cross_entropy_error(output : np.ndarray, desired : np.ndarray):
    '''
    交叉熵误差
    E = -∑k(t_k * log(y_k))
    '''
    delta = 1e-9
    if output.ndim == 1:
        output = output.reshape(1, output.size)
        desired = desired.reshape(1, desired.size)
    batch_size = desired.shape[0]
    return - np.sum(desired * np.log(output + delta)) / batch_size


def evaluate_loss(net : network, input : np.ndarray, desired : np.ndarray):
    '''
    计算一个网络的误差
    这里的 input 可以用 mini-batch 减小运算量
    input可以是二维的，第一个维度是第几个数据，第二个维度是向量的具体分量
    '''
    output = net.forward(input)
    return cross_entropy_error(output, desired)