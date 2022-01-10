from __future__ import annotations

from typing import Tuple, Sequence

import tensorflow as tf

from ..constants import BASE_COMPLEX_TYPE
from ..tensor_initialiser import TensorInitialiser
from ..custom_diag import custom_diag
from .abstract_tensor import AbstractTensor


class RawTensor(AbstractTensor):
    def __init__(self,
                 *,
                 name: str = None,
                 tensor: tf.Tensor = None,
                 shape: Tuple[int, ...] = None,
                 dtype=BASE_COMPLEX_TYPE,
                 init_method=None):
        super(RawTensor, self).__init__(name=name, dtype=dtype)

        # Check that the tensor content is provided XOR a shape for the tensor,
        # which will be randomly initialised
        assert ((tensor is None) ^ (shape is None))
        if tensor is not None:
            self._tensor = tensor
        else:
            self._tensor = TensorInitialiser(dtype=dtype, init_method=init_method)(shape)

    @property
    def shape(self):

        return self._tensor.shape

    @property
    def hidden_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def __str__(self) -> str:
        return (f'RawTensor {self._name}:\n'
                f'\t rank = {self.rank}\n'
                f'\tshape = {self.shape}')

    def amplitude(self, idx: tf.Tensor) -> tf.Tensor:

        return tf.gather(self._tensor, idx)

    def part_func(self) -> tf.Tensor:

        return tf.reduce_sum(self._tensor)

    def norm(self) -> tf.Tensor:

        return tf.norm(self._tensor)

    def swap_adjacent(self, fdx, tdx):
        assert abs(fdx - tdx) == 1

        shape = list(self.shape)
        shape[fdx], shape[tdx] = shape[tdx], shape[fdx]

        self._tensor = tf.transpose(self._tensor, shape)

    def merge_adjacent(self, fdx, tdx):
        assert abs(fdx - tdx) == 1

        shape = list(self.shape)
        shape[tdx] *= shape[fdx]
        shape.pop(fdx)

        self._tensor = tf.reshape(self._tensor, shape)

    @staticmethod
    def cut_phys_dim(*,
                     ltensor: RawTensor,
                     ldx: int,
                     rtensor: RawTensor,
                     rdx: int,
                     max_edge_dim=None,
                     **kwargs):
        raise NotImplementedError

    @staticmethod
    def contract_by_idx(*,
                        ltensor: RawTensor,
                        ldx: int,
                        rtensor: RawTensor,
                        rdx: int,
                        name: str = None,
                        **kwargs) -> RawTensor:
        assert ltensor._dtype == rtensor._dtype

        name = f'({ltensor._name}_{ldx}-{rtensor._name}_{rdx})' if name is None else name

        return RawTensor(name=name,
                         tensor=tf.tensordot(ltensor._tensor, rtensor._tensor, axes=[ldx, rdx]),
                         dtype=ltensor._dtype)

    @staticmethod
    def create_diag(*,
                    name: str = None,
                    rank: int = -1,
                    diag: tf.Tensor = None,
                    dtype=None) -> RawTensor:

        return RawTensor(name=name,
                         tensor=custom_diag(rank, diag, dtype=dtype),
                         dtype=dtype)
