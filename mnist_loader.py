import numpy as np
import gzip, struct

# 从 mnist/... .gz 中提取图片数据。

def load_images(path : str):
    '''
    加载 MNIST 图片数据集。返回的格式是一个 (60000, 28, 28) 张量，表示 [图片索引][x 坐标][y 坐标] -> 像素值。
    '''
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype = np.uint8).reshape(num, rows, cols)
    return data / 255

def load_images_flatten(path : str):
    '''
    加载 MNIST 图片数据集。返回的格式是一个 (60000, 784) 张量，表示 [图片索引][xy 坐标] -> 像素值。
    '''
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype = np.uint8).reshape(num, rows, cols)
    return data.reshape(num, -1) / 255

def load_labels(path):
    '''
    加载 MNIST 标签数据集（即每个图片对应数字的正确答案）。返回的格式是一个 (60000) 张量，表示 [图片索引] -> 答案。
    '''
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049
        data = np.frombuffer(f.read(), dtype = np.uint8)
    return data

def load_labels_to_vector(path):
    '''
    加载 MNIST 标签数据集，返回 one-hot 格式 (60000, 10)，
    即 [图片索引][类别] -> 1 或 0。
    '''
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return np.eye(10, dtype=np.uint8)[data]   # (60000, 10)

# 使用方法：
'''
data = load_images('mnist/train-images-idx3-ubyte.gz')
labels = load_labels('mnist/train-labels-idx1-ubyte.gz')
print(labels.shape)
'''