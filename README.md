# neural-network-101

只用 numpy 从零实现的全连接神经网络，在 MNIST 手写数字上训练与评估。

不依赖任何深度学习框架，前向传播、反向传播、softmax、交叉熵、数值梯度全部手写。

> **这是个人学习用的练习项目。**
>
> 目的是亲手实现一遍前向传播和反向传播来理解原理，不是可复用的库，也不是工程模板。
> 代码以可读性优先，没有做性能优化、数值稳定性加固、异常处理和单元测试。
> 如果要做正经的事情，请用 PyTorch / JAX 之类的框架。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `neural_network.py` | 网络主体：初始化、前向传播、反向传播、参数更新 |
| `backward.py` | 各算子的反向传播公式（ReLU / softmax+交叉熵 / 仿射层） |
| `loss_calculation.py` | 交叉熵误差与 loss 评估 |
| `numerical_gradient.py` | 中心差分数值梯度，用于校验反向传播 |
| `mnist_loader.py` | MNIST idx 格式解析（`gzip` + `struct`） |
| `train.py` | 训练 / 评估脚本 |
| `mnist_full_connect_1` | 训练好的权重（pickle），可直接加载 |
| `numpy_test_1.py`、`numpy.ipynb` | numpy 基础练习，与网络无关 |

## 数据集

**仓库里已附带 MNIST**，在 `mnist/` 下共四个文件（约 11 MB），克隆后无需额外下载即可直接运行：

```
mnist/train-images-idx3-ubyte.gz
mnist/train-labels-idx1-ubyte.gz
mnist/t10k-images-idx3-ubyte.gz
mnist/t10k-labels-idx1-ubyte.gz
```

数据来源：MNIST 数据集，由 Yann LeCun、Corinna Cortes 和 Christopher Burges 整理，
原发布页 <http://yann.lecun.com/exdb/mnist/>。此处仅作为学习用途随仓库附带。

## 运行

```bash
python train.py
```

`train.py` 的 `__main__` 里目前挂的是**梯度交叉校验**：在一个 `[784, 3, 10]` 的小网络上
用数值梯度核对反向传播，所以跑起来很快，也不产出模型。

要训练完整模型，把 `__main__` 换成：

```python
trainer = mnist_network_trainer("mnist_full_connect_1", [784, 100, 30, 20, 10])
trainer.load_mnist_train_dataset()
trainer.train()
trainer.save_training_data()
```

要在测试集上评估，换成 `load_mnist_examine_dataset()` + `examine()`。

## 实现要点

- 全连接层 + ReLU，输出层不加激活，直接接 softmax
- 参数按 `Y = f(XW + B)` 存储，`W` 形状为 `(fan_in, fan_out)`
- He 初始化：`W ~ N(0, sqrt(2 / fan_in))`
- loss 与梯度都是对 mini-batch 取均值
- 反向传播用中心差分数值梯度交叉校验，全局相对误差在 `1e-8` 量级
  （个别梯度接近 0 的分量逐元素相对误差会被放大到 `1e-6`，属正常现象）

## 结果

`[784, 100, 30, 20, 10]`，共 82360 个参数，mini-batch 100，
在 MNIST 测试集（10000 张）上准确率约 **96%**。

附带的是训练后的权重，`network.deserialize('mnist_full_connect_1')` 可直接加载。

## 参考

项目在整体思路、网络结构安排和代码组织上参考了：

- 斋藤康毅. 《深度学习入门：基于 Python 的理论与实现》（『ゼロから作る Deep Learning』）. 人民邮电出版社.
