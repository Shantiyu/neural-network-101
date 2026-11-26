from neural_network import network
from mnist_loader import *
import os
from numerical_gradient import *

class mnist_network_trainer:
    def __init__(self, model_name : str, model_size : list[int]):
        self.model_name = model_name
        if os.path.exists(model_name):  
            print("Model exist! loading.")
            self.network = network.deserialize(model_name)
        else:
            self.network = network(model_size)
            self.network.random_init()
            self.network.serialize(model_name)

        self.net_mut = copy.deepcopy(self.network)
        self.d_weight = copy.deepcopy(self.network.weight)
        self.d_shift = copy.deepcopy(self.network.shift)

    def save_training_data(self):
        self.network.serialize(self.model_name)
        print("Training data saved.")

    def load_mnist_train_dataset(self):
        self.data = load_images_flatten('mnist/train-images-idx3-ubyte.gz')
        self.labels = load_labels_to_vector('mnist/train-labels-idx1-ubyte.gz')
        self.training_data_size = self.data.shape[0]

    def load_mnist_examine_dataset(self):
        self.data = load_images_flatten('mnist/t10k-images-idx3-ubyte.gz')
        self.labels = load_labels_to_vector('mnist/t10k-labels-idx1-ubyte.gz')
        self.training_data_size = self.data.shape[0]

    def generate_mini_batch(self, batch_size = 100):
        batch_indexes = np.random.choice(self.training_data_size, batch_size, replace = False)
        self.minibatch_data = self.data[batch_indexes]
        self.minibatch_labels = self.labels[batch_indexes]

    # 用数值微分和反向传播梯度交叉比对检查二者算法正确性
    def backward_gradient_crosscheck(self):
        q = self.network.forward(self.minibatch_data, True)
        # 1 - based
        _, dl_dw_b, dl_db_b = self.network.backward(self.minibatch_labels)
        # 0 - based
        dl_dw_g, dl_db_g = numerical_gradient(self.minibatch_data, self.minibatch_labels, self.network, self.net_mut)
        def relative_error(analytic, numeric, eps = 1e-8):
            return np.max(np.abs(analytic - numeric) / (np.abs(analytic) + np.abs(numeric) + eps))
        for i in range(len(self.network.weight)):
            ew = relative_error(dl_dw_b[i + 1], dl_dw_g[i])
            eb = relative_error(dl_db_b[i + 1], dl_db_g[i])
            print(f"Layer {i}: dW relative error = {ew:.3e}   db relative error = {eb:.3e}")


    def step_train(self, step_k : float):
        # numerical_gradient(self.minibatch_data, self.minibatch_labels, self.network, self.net_mut, self.d_weight, self.d_shift)
        # for i in range(len(self.d_weight)):
        #     self.network.weight[i] -= step_k * self.d_weight[i]
        # for i in range(len(self.d_shift)):
        #     self.network.shift[i] -= step_k * self.d_shift[i]

        q = self.network.forward(self.minibatch_data, True)
        #loss = cross_entropy_error(q, self.minibatch_labels)
        dl_dz, dl_dw, dl_db = self.network.backward(self.minibatch_labels)
        self.network.adjust(step_k, dl_dw, dl_db)

    def train(self, step_count = 1000, step_k = 0.0001):
        print("Training start")
        for i in range(step_count):
            self.generate_mini_batch()
            self.step_train(step_k)
            if (i % 20 == 0):
                loss = evaluate_loss(self.network, self.minibatch_data, self.minibatch_labels)
                print(f"Training count {(i + 1)} | Loss : {loss}")

    def examine(self):
        print(f"Examining data set (count : {self.labels.shape[0]})")
        out = self.network.forward(self.data)
        pred = out.argmax(axis = 1)
        answer = self.labels.argmax(axis = 1)
        correct = (pred == answer)
        acc = correct.mean()
        print(f"Accuracy : {acc}")


if __name__ == '__main__':
    trainer = mnist_network_trainer("mnist_numeric_crosscheck", [784, 3, 10])
    # trainer.load_mnist_train_dataset()
    # trainer.train()
    # trainer.save_training_data()
    # trainer.examine()

    trainer.load_mnist_examine_dataset()
    trainer.generate_mini_batch()
    trainer.backward_gradient_crosscheck()

