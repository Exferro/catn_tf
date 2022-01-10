import logging

import numpy as np
import scipy as sp
import tensorflow as tf
import torch

from typing import Tuple, Callable

from .constants import BACKENDS
from .constants import DEFAULT_SVD_BACKEND, DEFAULT_BACKPROP_BACKEND
from .constants import DEFAULT_LORENTZIAN, DEFAULT_CUTOFF


def no_diff_svd(*,
                matrix: tf.Tensor,
                svd_backend: str = DEFAULT_SVD_BACKEND) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    assert len(matrix.shape) == 2
    assert svd_backend in BACKENDS

    if svd_backend == 'TF':
        s, u, v = tf.linalg.svd(matrix)
    elif svd_backend == 'TORCH':
        u, s, v_h = torch.linalg.svd(torch.from_numpy(matrix.numpy()), full_matrices=False)
        u = tf.constant(u.numpy(), dtype=matrix.dtype)
        s = tf.constant(s.numpy(), dtype=matrix.dtype)
        v = tf.linalg.adjoint(tf.constant(v_h.numpy(), dtype=matrix.dtype))
    elif svd_backend == 'SCIPY':
        u, s, v_h = sp.linalg.svd(matrix.numpy(), full_matrices=False)
        u = tf.constant(u, dtype=matrix.dtype)
        s = tf.constant(s, dtype=matrix.dtype)
        v = tf.linalg.adjoint(tf.constant(v_h, dtype=matrix.dtype))
    else:
        raise ValueError(f'Wrong backend {svd_backend} for the SVD. Choose one of {BACKENDS}')

    return u, tf.cast(s, dtype=matrix.dtype), v


def tf_svd_grad(u: tf.Tensor,
                s: tf.Tensor,
                v: tf.Tensor,
                du: tf.Tensor,
                ds: tf.Tensor,
                dv: tf.Tensor,
                lorentzian: float = DEFAULT_LORENTZIAN) -> tf.Tensor:
    v_h = tf.linalg.adjoint(v)
    s_square = tf.expand_dims(tf.multiply(s, s), axis=1)
    singular_diff = tf.subtract(tf.transpose(s_square), s_square)
    f = tf.divide(singular_diff, tf.add(tf.multiply(singular_diff, singular_diff), lorentzian))

    j = tf.multiply(f, tf.matmul(tf.linalg.adjoint(u), du))
    v_h_dv = tf.matmul(v_h, dv)
    k = tf.multiply(f, v_h_dv)
    l = tf.multiply(tf.eye(num_rows=v_h_dv.shape[0], dtype=v_h_dv.dtype), v_h_dv)

    s_matrix = tf.linalg.diag(s)
    s_inv = tf.divide(s_matrix, tf.add(tf.multiply(s_matrix, s_matrix), lorentzian))

    term_1 = tf.matmul(u, tf.matmul(tf.linalg.diag(ds), v_h))
    term_2 = tf.matmul(u, tf.matmul(tf.add(j, tf.linalg.adjoint(j)), tf.matmul(s_matrix, v_h)))
    term_3 = tf.matmul(u, tf.matmul(s_matrix, tf.matmul(tf.add(k, tf.linalg.adjoint(k)), v_h)))
    term_4 = tf.multiply(tf.constant(0.5, dtype=u.dtype),
                         tf.matmul(u, tf.matmul(s_inv, tf.matmul(tf.subtract(tf.linalg.adjoint(l), l), v_h))))

    return tf.add(term_1, tf.add(term_2, tf.add(term_3, term_4)))


def torch_h(x: torch.Tensor) -> torch.Tensor:
    return torch.conj(x.T)


def torch_svd_grad(u: tf.Tensor,
                   s: tf.Tensor,
                   v: tf.Tensor,
                   du: tf.Tensor,
                   ds: tf.Tensor,
                   dv: tf.Tensor,
                   lorentzian: float = DEFAULT_LORENTZIAN) -> tf.Tensor:
    u = torch.from_numpy(u.numpy())
    s = torch.from_numpy(s.numpy())
    v_h = torch.from_numpy(tf.linalg.adjoint(v).numpy())

    du = torch.from_numpy(du.numpy())
    ds = torch.from_numpy(ds.numpy())
    dv = torch.from_numpy(dv.numpy())

    s_square = torch.unsqueeze(torch.mul(s, s), dim=1)
    singular_diff = torch.sub(s_square.T, s_square)
    f = torch.div(singular_diff, torch.add(torch.mul(singular_diff, singular_diff), lorentzian))

    j = torch.mul(f, torch.matmul(torch_h(u), du))
    v_h_dv = torch.matmul(v_h, dv)
    k = torch.mul(f, v_h_dv)
    l = torch.mul(torch.eye(v_h_dv.shape[0], dtype=v_h_dv.dtype), v_h_dv)

    s_matrix = torch.diag(s)
    s_inv = torch.div(s_matrix, torch.add(torch.mul(s_matrix, s_matrix), lorentzian))

    term_1 = torch.matmul(u, torch.matmul(torch.diag(ds), v_h))
    term_2 = torch.matmul(u, torch.matmul(torch.add(j, torch_h(j)), torch.matmul(s_matrix, v_h)))
    term_3 = torch.matmul(u, torch.matmul(s_matrix, torch.matmul(torch.add(k, torch_h(k)), v_h)))
    term_4 = torch.mul(0.5, torch.matmul(u, torch.matmul(s_inv, torch.matmul(torch.sub(torch_h(l),
                                                                                       l),
                                                                             v_h))))

    return tf.constant(torch.add(term_1, torch.add(term_2, torch.add(term_3, term_4))))


def sp_h(x: np.ndarray) -> np.ndarray:
    return x.conj().T


def sp_svd_grad(u: tf.Tensor,
                s: tf.Tensor,
                v: tf.Tensor,
                du: tf.Tensor,
                ds: tf.Tensor,
                dv: tf.Tensor,
                lorentzian: float = DEFAULT_LORENTZIAN) -> tf.Tensor:
    u = u.numpy()
    s = s.numpy()
    v_h = sp_h(v.numpy())

    du = du.numpy()
    ds = ds.numpy()
    dv = dv.numpy()

    s_square = np.expand_dims(s * s, axis=1)
    singular_diff = s_square.T - s_square
    f = singular_diff / (singular_diff * singular_diff + lorentzian)

    j = f * (sp_h(u) @ du)
    v_h_dv = v_h @ dv
    k = f * v_h_dv
    l = np.eye(v_h_dv.shape[0], dtype=v_h_dv.dtype) * v_h_dv

    s_matrix = np.diag(s)
    s_inv = s_matrix / (s_matrix * s_matrix + lorentzian)

    term_1 = u @ (np.diag(ds) @ v_h)
    term_2 = u @ ((j + sp_h(j)) @ (s_matrix @ v_h))
    term_3 = u @ (s_matrix @ ((k + sp_h(k)) @ v_h))
    term_4 = 0.5 * (u @ (s_inv @ ((sp_h(l) - l) @ v_h)))

    return tf.constant(term_1 + term_2 + term_3 + term_4)


def svd(*,
        matrix: tf.Tensor,
        svd_backend: str = DEFAULT_SVD_BACKEND,
        backprop_backend: str = DEFAULT_BACKPROP_BACKEND) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Callable]:

    @tf.custom_gradient
    def inner_svd(matrix: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Callable]:
        assert backprop_backend in BACKENDS
        u, s, v = no_diff_svd(matrix=matrix, svd_backend=svd_backend)

        def grad(du: tf.Tensor, ds: tf.Tensor, dv: tf.Tensor) -> tf.Tensor:
            if backprop_backend == 'TF':
                da = tf_svd_grad(u, s, v, du, ds, dv)
            elif backprop_backend == 'TORCH':
                da = torch_svd_grad(u, s, v, du, ds, dv)
            elif backprop_backend == 'SCIPY':
                da = sp_svd_grad(u, s, v, du, ds, dv)
            else:
                raise ValueError(f'Wrong backend {backprop_backend} for the SVD. Choose one of {BACKENDS}')

            return da

        return (u, s, v), grad

    return inner_svd(matrix)


def trunc_svd(*,
              matrix: tf.Tensor = None,
              max_bond_dim: int = None,
              cutoff: float = DEFAULT_CUTOFF,
              svd_backend: str = DEFAULT_SVD_BACKEND,
              backprop_backend: str = DEFAULT_BACKPROP_BACKEND) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    u, s, v = svd(matrix=matrix, svd_backend=svd_backend, backprop_backend=backprop_backend)

    cutoff_dim = len(s.numpy()[s.numpy() > cutoff])
    new_bond_dim = min(max_bond_dim, cutoff_dim) if max_bond_dim is not None else cutoff_dim
    if new_bond_dim == 0:
        logger = logging.getLogger(f'nnqs.MPS')
        logger.warning(f'Zero new_bond_dim encountered during truncated_svd')
        new_bond_dim = 1

    s = s[:new_bond_dim]
    u = u[:, :new_bond_dim]
    v = v[:, :new_bond_dim]

    return u, s, v


def qr(*,
       matrix: tf.Tensor,
       svd_backend: str = DEFAULT_SVD_BACKEND,
       backprop_backend: str = DEFAULT_BACKPROP_BACKEND):
    u, s, v = svd(matrix=matrix,
                  svd_backend=svd_backend,
                  backprop_backend=backprop_backend)
    return u, tf.linalg.diag(s) @ tf.linalg.adjoint(v)
