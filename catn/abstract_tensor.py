from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from typing import Tuple

import numpy as np
import tensorflow as tf


class AbstractTensor(ABC):
    def __init__(self,
                 *,
                 name: str = None,
                 dtype):
        super(AbstractTensor, self).__init__()
        self._name = name
        self._dtype = dtype

    @property
    def name(self):

        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def rank(self) -> int:

        return len(self.shape)

    @property
    @abstractmethod
    def hidden_shape(self) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def amplitude(self, idx: tf.Tensor) -> tf.Tensor:
        ...

    @abstractmethod
    def part_func(self) -> tf.Tensor:
        ...

    @abstractmethod
    def norm(self) -> tf.Tensor:
        ...

    @staticmethod
    def calc_move_perm(rank, fdx, tdx):
        assert (0 <= fdx) and (fdx < rank)
        assert (0 <= tdx) and (tdx < rank)

        perm = np.arange(rank)
        if fdx < tdx:
            perm[fdx:tdx] = perm[(fdx + 1):(tdx + 1)]
        else:
            perm[(tdx + 1):(fdx + 1)] = perm[tdx:fdx]
        perm[tdx] = fdx

        return perm

    @abstractmethod
    def swap_adjacent(self, fdx, tdx):
        ...

    def move(self, fdx, tdx):
        assert (0 <= fdx) and (fdx < self.rank)
        assert (0 <= tdx) and (tdx < self.rank)

        if fdx < tdx:
            for idx in range(fdx, tdx):
                self.swap_adjacent(idx, idx + 1)
        elif fdx > tdx:
            for idx in range(fdx, tdx, -1):
                self.swap_adjacent(idx, idx - 1)
        else:
            pass

        #return AbstractTensor.calc_move_perm(self._visible_num, fdx, tdx)

    @abstractmethod
    def merge_adjacent(self, fdx, tdx):
        ...

    def merge(self, fdx: int, tdx: int, cross: bool = False):
        assert (0 <= fdx) and (fdx < self.rank)
        assert (0 <= tdx) and (tdx < self.rank)
        assert fdx != tdx

        if fdx < tdx:
            move_to = tdx - 1
        else:
            move_to = tdx + 1
        self.move(fdx, move_to)
        if cross:
            self.swap_adjacent(move_to, tdx)
        self.merge_adjacent(move_to, tdx)

    @staticmethod
    @abstractmethod
    def cut_phys_dim(*,
                     ltensor: AbstractTensor,
                     ldx: int,
                     rtensor: AbstractTensor,
                     rdx: int,
                     max_edge_dim=None,
                     **kwargs):
        ...

    @staticmethod
    @abstractmethod
    def contract_by_idx(*,
                        ltensor: AbstractTensor,
                        ldx: int,
                        rtensor: AbstractTensor,
                        rdx: int,
                        name: str = None,
                        **kwargs) -> AbstractTensor:
        """
        Basically a tensordot operation but with only one axis involved
        from each tensor

        :param ltensor:
        :param ldx:
        :param rtensor:
        :param rdx:
        :param name:
        :param kwargs:
        :return:
        """
        ...

    @staticmethod
    @abstractmethod
    def create_diag(*,
                    name: str = None,
                    rank: int = -1,
                    diag: tf.Tensor = None,
                    dtype=None) -> AbstractTensor:
        ...
