from neural_network import network
from loss_calculation import *
import numpy as np
import copy

def numerical_gradient(
          input : np.ndarray, desired : np.ndarray, net : network,
          net_mut : network, delta = 1e-5):

    def loss():
        return evaluate_loss(net_mut, input, desired)

    d_weight = [np.zeros_like(w) for w in net.weight]
    d_shift  = [np.zeros_like(b) for b in net.shift]

    for i in range(len(net.weight)):
        shape = net_mut.weight[i].shape
        for x in range(shape[0]):
            print(f"[Numeric Grad.] checking W : {x} of {shape[0]}")
            for y in range(shape[1]):
                net_mut.weight[i] [x][y] = net.weight[i] [x][y] - delta
                loss_1 = loss()
                net_mut.weight[i] [x][y] = net.weight[i] [x][y] + delta
                loss_2 = loss()
                net_mut.weight[i] [x][y] = net.weight[i] [x][y]
                derivative = (loss_2 - loss_1) / delta / 2
                d_weight[i] [x][y] = derivative

    for i in range(len(net.shift)):
            shape = net_mut.shift[i].shape
            for x in range(shape[0]):
                print(f"[Numeric Grad.] checking B : {x} of {shape[0]}")
                net_mut.shift[i] [x] = net.shift[i] [x] - delta
                loss_1 = loss()
                net_mut.shift[i] [x] = net.shift[i] [x] + delta
                loss_2 = loss()
                net_mut.shift[i] [x] = net.shift[i] [x]
                derivative = (loss_2 - loss_1) / delta / 2
                d_shift[i] [x] = derivative

    return (d_weight, d_shift)