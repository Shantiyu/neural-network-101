from neural_network import network
from layers_affine import *
from layers_convolution import *
from layer_util import *
from mnist_loader import *
import matplotlib.pyplot as plt
import os
from pathlib import Path

class mnist_network_trainer:
    def __init__(self, model_name : str, model_layers : layer_sequence):
        self.model_name = model_name
        self.loss_cache_path = Path(f"loss_cache_{Path(model_name).stem}.txt")
        if os.path.exists(model_name):  
            print("Model exist! loading.")
            self.network = network.deserialize(model_name)
        else:
            self.network = network(model_layers)
            # 不在构造时保存，避免训练中断后留下一个"看起来存在"的未训练模型

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

    def step_train(self, input : np.ndarray, labels : np.ndarray, step_k : float) -> float:
        return self.network.step_train(input, labels, step_k)

    @staticmethod
    def _last_loss_index(path: Path) -> int:
        """读取已有 cache 的最后一个 index；空文件或不存在时返回 0。"""
        if not path.exists():
            return 0

        last_index = 0
        with open(path, "r", encoding="utf-8") as cache:
            for line in cache:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    last_index = int(line.split(",", 1)[0])
                except ValueError:
                    continue
        return last_index

    def train(self, step_count = 1000, step_k = 0.03, show_loss_plot=False):
        print("Training start")
        start_index = self._last_loss_index(self.loss_cache_path)
        mode = "a" if start_index > 0 else "w"
        if start_index > 0:
            print(f"Appending to {self.loss_cache_path} from index {start_index + 1}")

        loss_list = []
        with open(self.loss_cache_path, mode, encoding="utf-8") as loss_file:
            if mode == "w":
                loss_file.write(f"# model={self.model_name}, lr={step_k}\n")
                loss_file.write("# index,loss\n")
            else:
                loss_file.write(f"# resume lr={step_k}\n")

            for i in range(step_count):
                self.generate_mini_batch()
                loss = self.step_train(self.minibatch_data, self.minibatch_labels, step_k)
                loss_list.append(loss)
                index = start_index + i + 1
                loss_file.write(f"{index},{loss:.8f}\n")
                loss_file.flush()
                print(f"Training count {index} | Loss : {loss}")
                if i % 100 == 0:
                    self.network.serialize(self.model_name)
                    print("Checkpoint saved.")
        print(f"Loss cache saved to {self.loss_cache_path}")
        if show_loss_plot:
            self.plot_loss(loss_list, start_index)

    def plot_loss(self, loss_list, start_index=0):
        plt.figure(figsize = (10, 5))
        steps = range(start_index + 1, start_index + len(loss_list) + 1)
        plt.plot(steps, loss_list, color = 'tab:blue', linewidth = 1.5)
        plt.title(f"Training Loss - {self.model_name}")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(True, linestyle = '--', alpha = 0.5)
        plt.tight_layout()
        plt.show()

    def examine(self):
        print(f"Examining data set (count : {self.labels.shape[0]})")
        out = self.network.forward(self.data)
        pred = out.argmax(axis = 1)
        answer = self.labels.argmax(axis = 1)
        correct = (pred == answer)
        acc = correct.mean()
        print(f"Accuracy : {acc}")

if __name__ == '__main__':
    layers = layer_sequence(
        reshape(1, 28, 28),                      # (N, 784) -> (N, 1, 28, 28) # type: ignore

        convolution(32, 1, 28, 28, 3, 3, 1, 1),  # -> (N, 32, 28, 28)
        relu(),
        convolution(32, 32, 28, 28, 3, 3, 1, 1),
        relu(),
        pooling_max(2, 2, 2),                    # -> (N, 32, 14, 14)

        convolution(64, 32, 14, 14, 3, 3, 1, 1),
        relu(),
        convolution(64, 64, 14, 14, 3, 3, 1, 1),
        relu(),
        pooling_max(2, 2, 2),                    # -> (N, 64, 7, 7)

        convolution(128, 64, 7, 7, 3, 3, 1, 1),
        relu(),
        pooling_max(2, 2, 2),                    # -> (N, 128, 3, 3)

        flatten(),                               # -> (N, 128 * 3 * 3) = (N, 1152)
        affine(1152, 128),
        relu(),
        affine(128, 10),
        softmax_with_cee(),
    )
    trainer = mnist_network_trainer("models/mnist_conv_02", layers)
    trainer.load_mnist_train_dataset()
    trainer.train(1000, 0.03, True)
    trainer.save_training_data()

    trainer.load_mnist_examine_dataset()
    trainer.examine()

# layers = layer_sequence(
#         affine(784, 500),
#         relu(),
#         affine(500, 10),
#         softmax_with_cee()
#     )


