import numpy as np

def relu_dl_dy(dl_dz : np.ndarray, y : np.ndarray) -> np.ndarray:
    return dl_dz * (y > 0)

def sigmoid_dl_dy(dl_dz : np.ndarray, z : np.ndarray) -> np.ndarray:
    return dl_dz * z * (1 - z)

def softmax_cee_dl_dz(q : np.ndarray, t : np.ndarray) -> np.ndarray:
    r = q.shape[0]
    return (q - t) / r

def affine_dl_dw(z : np.ndarray, dl_dy : np.ndarray) -> np.ndarray:
    return z.T @ dl_dy

def affine_dl_dz(w : np.ndarray, dl_dy : np.ndarray) -> np.ndarray:
    return dl_dy @ w.T

def affine_dl_db(dl_dy : np.ndarray) -> np.ndarray:
    return dl_dy.sum(axis=0)
