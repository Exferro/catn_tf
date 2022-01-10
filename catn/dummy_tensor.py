from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from ..constants import BASE_COMPLEX_TYPE
from .abstract_tensor import AbstractTensor


class DummyTensor(AbstractTensor):
    def __init__(self,
                 *,
                 name: str = None,
                 shape: Tuple[int],
                 value=-1.0,
                 dtype=BASE_COMPLEX_TYPE):
        super(DummyTensor, self).__init__(name=name, dtype=dtype)
        self._shape = tuple(shape)
        self._value = value

    @property
    def shape(self):

        return self._shape

    @property
    def hidden_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def __str__(self):
        return (f'DummyTensor {self._name}:\n'                
                f'\t rank = {self.rank}\n'
                f'\tshape = {self.shape}')

    def amplitude(self, idx: tf.Tensor) -> tf.Tensor:

        return tf.fill(idx.shape, self._value)

    def part_func(self) -> tf.Tensor:

        return tf.reduce_prod(self._shape) * self._value

    def norm(self) -> tf.Tensor:

        return tf.reduce_prod(self._shape) * tf.norm(self._value)

    def swap_adjacent(self, fdx, tdx):
        assert abs(fdx - tdx) == 1

        shape = list(self.shape)
        shape[fdx], shape[tdx] = shape[tdx], shape[fdx]

        self._shape = tuple(shape)

    def merge_adjacent(self, fdx, tdx):
        assert abs(fdx - tdx) == 1
        shape = list(self.shape)
        shape[tdx] *= shape[fdx]
        shape.pop(fdx)

        self._shape = tuple(shape)

    @staticmethod
    def cut_phys_dim(*,
                     ltensor: DummyTensor,
                     ldx: int,
                     rtensor: DummyTensor,
                     rdx: int,
                     max_edge_dim=None,
                     **kwargs):
        pass

    @staticmethod
    def contract_by_idx(*,
                        ltensor: DummyTensor,
                        ldx: int,
                        rtensor: DummyTensor,
                        rdx: int,
                        name: str = None,
                        **kwargs) -> DummyTensor:
        assert ltensor._dtype == rtensor._dtype

        name = f'({ltensor._name}_{ldx}-{rtensor._name}_{rdx})' if name is None else name
        ltensor.move(ldx, ltensor.rank - 1)
        rtensor.move(rdx, 0)

        return DummyTensor(name=name,
                           shape=tuple(ltensor.shape[:-1] + rtensor.shape[1:]),
                           dtype=ltensor._dtype)

    @staticmethod
    def create_diag(*,
                    name: str = None,
                    rank: int = -1,
                    diag: tf.Tensor = None,
                    dtype=None) -> DummyTensor:
        raise NotImplementedError
