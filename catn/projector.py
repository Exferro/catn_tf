import numpy as np
import scipy as sp
import tensorflow as tf

from typing import Tuple, Union

from scipy.optimize import minimize

from ..tensor_initialiser import TensorInitialiser


class Projector:
    def __init__(self,
                 *,
                 a_full: tf.Tensor = None,
                 b_full: tf.Tensor = None,
                 method: str = 'CG'):
        assert a_full.shape[1] == b_full.shape[0]
        assert a_full.dtype == b_full.dtype

        self._a_full = a_full
        self._b_full = b_full
        self._a_full_adj = tf.linalg.adjoint(a_full)
        self._b_full_adj = tf.linalg.adjoint(b_full)
        self._dtype = self._a_full.dtype

        self._a_dim = self._a_full.shape[0]
        self._b_dim = self._b_full.shape[1]

        self._method = method

    def x_to_a_b(self,
                 *,
                 x: Union[tf.Tensor, np.ndarray]  = None,
                 bond_dim: int = None) -> Tuple[tf.Tensor, tf.Tensor]:
        a, b = tf.split(x, [self._a_dim * bond_dim, self._b_dim * bond_dim])
        return tf.reshape(a, (self._a_dim, bond_dim)), tf.reshape(b, (bond_dim, self._b_dim))

    def a_b_to_x(self,
                 *,
                 a: tf.Tensor = None,
                 b: tf.Tensor = None) -> np.ndarray:
        return tf.concat([tf.reshape(a, (-1, )), tf.reshape(b, (-1, ))], axis=0).numpy()

    def error_norm(self, *, a_trunc: tf.Tensor = None, b_trunc: tf.Tensor = None):
        a_trunc_adj = tf.linalg.adjoint(a_trunc)
        b_trunc_adj = tf.linalg.adjoint(b_trunc)

        return tf.sqrt(tf.einsum('km,mi,ij,jk', b_trunc_adj, a_trunc_adj, a_trunc, b_trunc)
                       - tf.einsum('km,mi,il,lk', b_trunc_adj, a_trunc_adj, self._a_full, self._b_full)
                       - tf.einsum('kn,ni,ij,jk', self._b_full_adj, self._a_full_adj, a_trunc, b_trunc)
                       + tf.einsum('kn,ni,il,lk', self._b_full_adj, self._a_full_adj, self._a_full, self._b_full))

    def generate_fun(self, bond_dim: int = None):
        def fun(x):
            a_trunc, b_trunc = self.x_to_a_b(x=x, bond_dim=bond_dim)
            loss = self.error_norm(a_trunc=a_trunc, b_trunc=b_trunc)
            print(f'Projecting {self._a_full.shape}, {self._b_full.shape} onto '
                  f'max_phys_dim = {bond_dim}, loss = {loss}')
            return loss.numpy()
        return fun

    def generate_jac(self, bond_dim: int = None):
        def jac(x):
            a_trunc, b_trunc = self.x_to_a_b(x=x, bond_dim=bond_dim)
            a_trunc_adj = tf.linalg.adjoint(a_trunc)
            b_trunc_adj = tf.linalg.adjoint(b_trunc)

            error_norm = self.error_norm(a_trunc=a_trunc, b_trunc=b_trunc)
            print(f'In jac: projecting {self._a_full.shape}, {self._b_full.shape} onto '
                  f'max_phys_dim = {bond_dim}, loss = {error_norm}')

            a_grad = (2 / error_norm) * (tf.matmul(a_trunc, tf.matmul(b_trunc, b_trunc_adj))
                                         - tf.matmul(self._a_full, tf.matmul(self._b_full, b_trunc_adj)))
            b_grad = (2 / error_norm) * (tf.matmul(tf.matmul(a_trunc_adj, a_trunc), b_trunc)
                                         - tf.matmul(tf.matmul(a_trunc_adj, self._a_full), self._b_full))

            return self.a_b_to_x(a=a_grad, b=b_grad)

        return jac

    # def project(self,
    #             *,
    #             max_bond_dim: int = None,
    #             tol: float = None,
    #             cutoff: float = None) -> Tuple[tf.Tensor, tf.Tensor]:
    #     start_x = TensorInitialiser(dtype=self._dtype)(shape=((self._a_dim + self._b_dim) * max_bond_dim,))
    #
    #     result = minimize(fun=self.generate_fun(bond_dim=max_bond_dim),
    #                       x0=start_x,
    #                       method=self._method,
    #                       jac=self.generate_jac(bond_dim=max_bond_dim),
    #                       tol=tol)
    #
    #     a, b = self.x_to_a_b(x=result.x, bond_dim=max_bond_dim)
    #     # print(f'Project of {self._a_full.shape} and {self._b_full.shape} gives '
    #     #       f'{a.shape} and {b.shape}')
    #     u_a, s_a, v_h_a = sp.linalg.svd(a, full_matrices=False)
    #     u_b, s_b, v_h_b = sp.linalg.svd(b, full_matrices=False)
    #
    #     bond_matrix = np.diag(s_a) @ v_h_a @ u_b @ np.diag(s_b)
    #     u, s, v_h = sp.linalg.svd(bond_matrix)
    #     # print(f'Further truncation singular values: {s}')
    #     cutoff_dim = len(s[s > cutoff])
    #     new_dim = min(max_bond_dim, cutoff_dim)
    #     u = u[:, :new_dim]
    #     sqrt_s = np.sqrt(np.diag(s[:new_dim]))
    #     v_h = v_h[:new_dim, :]
    #     a_trunc, b_trunc = tf.constant(u_a @ u @ sqrt_s), tf.constant(sqrt_s @ v_h @ v_h_b)
    #     # print(f'Further truncation gives {a_trunc.shape} and {b_trunc.shape}')
    #
    #     a_smart, b_smart = self.project_smart(max_bond_dim=max_bond_dim, cutoff=cutoff)
    #     # print(f'At the same time smart truncation gives {a_smart.shape} and {b_smart.shape}')
    #     # print(f'Difference between project and smart SVD: {tf.norm(a_trunc @ b_trunc - a_smart @ b_smart)}')
    #     # print(f'Difference between project and ground truth: {tf.norm(a_trunc @ b_trunc - self._a_full @ self._b_full)}')
    #     # print(f'Difference between smart SVD and ground truth: {tf.norm(a_smart @ b_smart - self._a_full @ self._b_full)}\n')
    #
    #     return a_trunc, b_trunc
        #
        # return self.x_to_a_b(x=result.x, bond_dim=max_bond_dim)

    def project(self,
                *,
                max_bond_dim: int = None,
                cutoff: float = None,
                tol: float = None) -> Tuple[tf.Tensor, tf.Tensor]:
        u_a, s_a, v_h_a = sp.linalg.svd(self._a_full.numpy(), full_matrices=False)
        u_b, s_b, v_h_b = sp.linalg.svd(self._b_full.numpy(), full_matrices=False)

        bond_matrix = np.diag(s_a) @ v_h_a @ u_b @ np.diag(s_b)
        u, s, v_h = sp.linalg.svd(bond_matrix)
        # print(f'Smart truncation singular values: {s}')
        cutoff_dim = len(s[s > cutoff])
        new_dim = min(max_bond_dim, cutoff_dim)
        if new_dim == 0:
            print(f'Zero new_bond_dim encountered during projection')
            print(s)
            new_dim = max_bond_dim
        u = u[:, :new_dim]
        sqrt_s = np.sqrt(np.diag(s[:new_dim]))
        v_h = v_h[:new_dim, :]

        return tf.constant(u_a @ u @ sqrt_s), tf.constant(sqrt_s @ v_h @ v_h_b)
