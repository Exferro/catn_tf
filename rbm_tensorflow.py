import time

import numpy as np
import scipy as sp
import tensorflow as tf
import torch

from .particle_enum import Particle
from .constants import BASE_COMPLEX_TYPE, BASE_INT_TYPE
from .tensor_initialiser import TensorInitialiser

from .picklable import Picklable

from .catn.constants import DEFAULT_CUTOFF, DEFAULT_MAX_PHYS_DIM, DEFAULT_MAX_BOND_DIM
from .catn import MPS, Node
from .catn.tensor_network import AbstractTensorNetwork
from .catn.tensor_network import BaseTensorNetwork


class RBM(Picklable):
    def __init__(self,
                 *,
                 name: str = None,
                 particle: Particle = Particle.qubit,
                 visible_num: int,
                 hidden_num: int,
                 dtype=BASE_COMPLEX_TYPE,
                 idx_dtype=BASE_INT_TYPE,
                 vb: tf.Tensor = None,
                 hb: tf.Tensor = None,
                 wm: tf.Tensor = None,
                 init_method=None):
        super(RBM, self).__init__()
        self._name = name
        self._particle = particle

        self._visible_num = visible_num
        self._hidden_num = hidden_num

        self._dtype = dtype
        self._idx_dtype = idx_dtype

        tensor_initialiser = TensorInitialiser(dtype=self._dtype,
                                               init_method=init_method)
        if vb is None:
            self._vb = tensor_initialiser((visible_num, ))
        else:
            assert vb.shape == (visible_num, )
            assert vb.dtype == self._dtype
            self._vb = vb
        if hb is None:
            self._hb = tensor_initialiser((hidden_num, ))
        else:
            assert hb.shape == (hidden_num, )
            assert hb.dtype == self._dtype
            self._hb = hb
        if wm is None:
            self._wm = tensor_initialiser((hidden_num, visible_num))
        else:
            assert wm.shape == (hidden_num, visible_num)
            assert wm.dtype == self._dtype
            self._wm = wm

        # Constants required for conversion to custom TensorNetwork
        self._empty_diag = tf.constant([1.0, 0.0], dtype=self._dtype)
        self._empty_ones = tf.constant([[1.0, 1.0], [1.0, 0.0]], dtype=self._dtype)

    def __str__(self):
        return (f'RBM {self._name}:\n'
                f'\tvisible_num: {self._visible_num}\n'
                f'\thidden_num: {self._hidden_num}\n'
                f'\tdtype: {self._dtype}\n'
                f'\tidx_dtype: {self._idx_dtype}\n'
                f'\tvb: {self._vb}\n'
                f'\thb: {self._hb}\n'
                f'\twm: {self._wm}\n')

    def idx_to_visible(self, idx):
        visible = tf.cast(tf.reshape(tf.math.floormod(tf.bitwise.right_shift(tf.reshape(idx, (-1, 1)),
                                                                             tf.range(self._visible_num,
                                                                                      dtype=self._idx_dtype)),
                                                      2),
                                     (-1, self._visible_num)),
                          dtype=self._idx_dtype)
        if self._particle == Particle.qubit:
            return visible
        else:
            return 1 - 2 * visible

    def visible_to_idx(self, visible: tf.Tensor) -> tf.Tensor:
        if not tf.is_tensor(visible):
            visible = tf.constant(visible)
        if visible.dtype != self._idx_dtype:
            visible = tf.cast(visible, dtype=self._idx_dtype)
        if self._particle == Particle.spin:
            visible = (1 - visible) // 2
        two_powers = 2 ** tf.range(self._visible_num, dtype=self._idx_dtype)

        return tf.tensordot(visible, two_powers, axes=1)

    def amplitude(self, visible_state):
        assert visible_state.shape[-1] == self._visible_num
        if len(visible_state.shape) == 1:
            visible_state = tf.expand_dims(visible_state, axis=0)
        if visible_state.dtype != self._dtype:
            visible_state = tf.cast(visible_state, dtype=self._dtype)

        if self._particle == Particle.qubit:
            return tf.multiply(tf.exp(tf.linalg.matvec(visible_state,
                                                       self._vb)),
                               tf.reduce_prod(tf.add(1.0, tf.exp((tf.add(tf.tensordot(visible_state,
                                                                                      self._wm,
                                                                                      axes=[-1, -1]),
                                                                         self._hb)))),
                                              axis=-1))
        elif self._particle == Particle.spin:
            return tf.multiply(tf.exp(tf.linalg.matvec(visible_state,
                                                       self._vb)),
                               tf.reduce_prod(2 * tf.cosh((tf.add(tf.tensordot(visible_state,
                                                                               self._wm,
                                                                               axes=[-1, -1]),
                                                                  self._hb))),
                                              axis=-1))

    def part_func_bf(self, **kwargs):
        start_time = time.time()
        part_func = tf.reduce_sum(self.amplitude(self.idx_to_visible(tf.range(2 ** self._visible_num,
                                                                              dtype=self._idx_dtype))))
        run_time = time.time() - start_time

        return part_func, run_time

    def calc_bias_diag(self,
                       *,
                       bias_vector=None,
                       pos=None):
        if self._particle == Particle.qubit:
            result = self._empty_diag + tf.scatter_nd([[1]], [tf.exp(bias_vector[pos])], (2,))
        else:
            result = tf.scatter_nd([[0], [1]],
                                   [tf.exp(bias_vector[pos]), tf.exp(-bias_vector[pos])],
                                   (2,))

        return result

    def calc_weight_tensor(self,
                           *,
                           pos_0=None,
                           pos_1=None):
        if self._particle == Particle.qubit:
            result = self._empty_ones + tf.scatter_nd([[1, 1]],
                                                      [tf.exp(self._wm[pos_0, pos_1])], (2, 2))
        else:
            result = tf.scatter_nd([[0, 0], [0, 1], [1, 0], [1, 1]],
                                   [tf.exp(self._wm[pos_0, pos_1]),
                                      tf.exp(-self._wm[pos_0, pos_1]),
                                      tf.exp(-self._wm[pos_0, pos_1]),
                                      tf.exp(self._wm[pos_0, pos_1])], (2, 2))

        return result

    def to_tensor_network(self,
                          *,
                          tensor_network_class=None,
                          max_phys_dim: int = DEFAULT_MAX_PHYS_DIM,
                          max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                          weights_to: str ='visible',
                          cutoff: float = DEFAULT_CUTOFF,
                          minimise_swaps: bool = False,
                          **kwargs) -> AbstractTensorNetwork:
        assert weights_to in ('visible', 'hidden')
        tensor_network = tensor_network_class(name=self._name,
                                              max_phys_dim=max_phys_dim,
                                              minimise_swaps=minimise_swaps)

        for idx in range(self._visible_num):
            cur_diag = self.calc_bias_diag(bias_vector=self._vb, pos=idx)
            cur_mps = MPS.create_diag(name=f'v_{idx}',
                                      rank=self._hidden_num,
                                      diag=cur_diag,
                                      dtype=self._dtype,
                                      max_bond_dim=max_bond_dim,
                                      cutoff=cutoff)
            if weights_to == 'visible':
                for jdx in range(self._hidden_num):
                    weight_tensor = self.calc_weight_tensor(pos_0=jdx, pos_1=idx)
                    cur_mps._tensors[jdx] = tf.einsum('iaj,ab->ibj',
                                                      cur_mps._tensors[jdx],
                                                      weight_tensor)
            tensor_network.add_node(Node(name=f'v_{idx}',
                                         tensor=cur_mps))
        for jdx in range(self._hidden_num):
            cur_diag = self.calc_bias_diag(bias_vector=self._hb, pos=jdx)
            cur_mps = MPS.create_diag(name=f'h_{jdx}',
                                      rank=self._visible_num,
                                      diag=cur_diag,
                                      dtype=self._dtype,
                                      max_bond_dim=max_bond_dim,
                                      cutoff=cutoff)
            if weights_to == 'hidden':
                for idx in range(self._visible_num):
                    weight_tensor = self.calc_weight_tensor(pos_0=jdx, pos_1=idx)
                    cur_mps._tensors[idx] = tf.einsum('ab,ibj->iaj',
                                                      weight_tensor,
                                                      cur_mps._tensors[idx])
            tensor_network.add_node(Node(name=f'h_{jdx}',
                                         tensor=cur_mps))

        for idx in range(self._visible_num):
            for jdx in range(self._hidden_num):
                tensor_network.connect_by_name(f'v_{idx}', jdx, f'h_{jdx}', (idx + jdx) % self._visible_num)
        return tensor_network

    def calc_bias_diag_full(self,
                            *,
                            rank: int = None,
                            bias_vector: tf.Tensor = None,
                            pos: int = None):
        diag_shape = tuple([2] * rank)
        all_zeros = tuple([0] * rank)
        all_ones = tuple([1] * rank)
        result = np.zeros(shape=diag_shape, dtype=bias_vector.numpy().dtype)

        if self._particle == Particle.qubit:
            result[all_zeros] = 1.0
            result[all_ones] = np.exp(bias_vector[pos].numpy())
        else:
            result[all_zeros] = tf.exp(bias_vector[pos])
            result[all_ones] = tf.exp(-bias_vector[pos])

        return tf.constant(result, dtype=bias_vector.dtype)

    def part_func_tf_slow(self, **kwargs):
        MPS.OPERATION_MODE = 'SLOW'
        MPS.EXT_CONTRACTOR = 'tf_slow'
        tensor_network = self.to_tensor_network(tensor_network_class=BaseTensorNetwork,
                                                **kwargs)
        start_time = time.time()
        part_func = tensor_network.contract()
        run_time = time.time() - start_time

        return part_func, run_time

    def part_func_tf_fast(self, **kwargs):
        MPS.OPERATION_MODE = 'FAST'
        MPS.EXT_CONTRACTOR = 'tf_fast'
        tensor_network = self.to_tensor_network(tensor_network_class=BaseTensorNetwork,
                                                **kwargs)
        start_time = time.time()
        part_func = tensor_network.contract()
        run_time = time.time() - start_time

        return part_func, run_time

