# neural-network-101

只用 numpy 从零实现的全连接神经网络，在 MNIST 手写数字上训练与评估。

不依赖任何深度学习框架，前向传播、反向传播、softmax、交叉熵、数值梯度全部手写。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `neural_network.py` | 网络主体：初始化、前向传播、反向传播、参数更新 |
| `backward.py` | 各算子的反向传播公式（ReLU / softmax+交叉熵 / 仿射层） |
| `loss_calculation.py` | 交叉熵误差与 loss 评估 |
| `numerical_gradient.py` | 中心差分数值梯度，用于校验反向传播 |
| `mnist_loader.py` | MNIST idx 格式解析（`gzip` + `struct`） |
| `train.py` | 训练 / 评估脚本 |
| `numpy_test_1.py`、`numpy.ipynb` | numpy 基础练习，与网络无关 |

## 准备数据

数据集不进版本库，需要自己下载解压后的四个文件放到 `mnist/`：

```
mnist/train-images-idx3-ubyte.gz
mnist/train-labels-idx1-ubyte.gz
mnist/t10k-images-idx3-ubyte.gz
mnist/t10k-labels-idx1-ubyte.gz
```

来源：<http://yann.lecun.com/exdb/mnist/>（或任意 MNIST 镜像）。

## 运行

```bash
python train.py
```

## 实现要点

- 全连接层 + ReLU，输出层不加激活，直接接 softmax
- 参数按 `Y = f(XW + B)` 存储，`W` 形状为 `(fan_in, fan_out)`
- He 初始化：`W ~ N(0, sqrt(2 / fan_in))`
- loss 与梯度都是对 mini-batch 取均值
- 反向传播用数值梯度交叉校验，全局相对误差在 `1e-8` 量级

## 结果

`[784, 100, 30, 20, 10]`，mini-batch 100，在 MNIST 测试集上准确率约 **96%**。

## 引用

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
