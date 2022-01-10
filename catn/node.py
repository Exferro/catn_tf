from __future__ import annotations

import logging

import tensorflow as tf

from typing import Dict, Union, Tuple

from ..logging import Logging
from .abstract_tensor import AbstractTensor
from .mps import MPS


class Node(Logging):
    def __init__(self,
                 name,
                 *,
                 id: int = None,
                 tensor: AbstractTensor = None,
                 neighbours: Dict[int, Union[int, Tuple[int, ...]]] = None):
        super(Node, self).__init__()
        assert isinstance(name, str)
        self._name = name

        self._id = id
        self._tensor = tensor
        if self._tensor.name is None:
            self._tensor.name = f'{self._name}'

        self._neighbours = neighbours if neighbours is not None else dict()

    def __str__(self):
        return (f'Node {self._name}, ID = {self._id}:\n'
                f'\ttensor = {self._tensor}\n'
                f'\tneighbours = {self._neighbours}\n')

    @property
    def name(self):

        return self._name

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value: int):

        self._id = value

    @property
    def shape(self):
        return self._tensor.shape


    @property
    def rank(self):

        return self._tensor.rank

    @property
    def hidden_shape(self):
        return self._tensor.hidden_shape

    @property
    def neighbours(self):
        return self._neighbours

    def part_func(self):

        return self._tensor.part_func()

    def norm(self):

        return self._tensor.norm()

    def add_neighbour(self,
                     *,
                     nid: int = None,
                     idx: int = None):
        assert nid is not None
        assert idx is not None
        assert 0 <= idx
        assert idx < self.rank

        if nid not in self._neighbours:
            self._neighbours[nid] = tuple()
        if idx in self._neighbours[nid]:
            raise ValueError(f'Index {idx} was already added as a reference to '
                             f'the neighbouring Node {nid} of the Node {self._name}')
        self._neighbours[nid] = self._neighbours[nid] + (idx,)

    @staticmethod
    def gated_offset(x, *, gate=None):

        return x - 1 if x > gate else x

    def merge_duplicate(self, dup_nid: int, cross: bool = False):
        """
        Used to merge duplicate physical indices corresponding to the neighbour with 
        ID = dup_nid
        Assumes that the neighbour ID corresponding to the new physical index
        is the one referring to tdx, not fdx (i.e. after the merge there won't be
        fdx in self._neighbours). Rearranges all other physical indices properly.

        :param dup_nid:
        :param cross:
        :return:
        """
        assert len(self._neighbours[dup_nid]) == 2
        fdx = self._neighbours[dup_nid][0]
        tdx = self._neighbours[dup_nid][1]

        fdx, tdx = max(fdx, tdx), min(fdx, tdx)

        self._logger.debug(f'Node #{self.id} has duplicates with Node #{dup_nid}')
        self._logger.debug(f'fdx = {fdx}, tdx = {tdx}, shape before = {self.shape}')
        for nid in self._neighbours:
            self._neighbours[nid] = tuple(Node.gated_offset(x, gate=fdx) for x in self._neighbours[nid])
        self._neighbours[dup_nid] = (Node.gated_offset(tdx, gate=fdx),)

        self._tensor.merge(fdx, tdx, cross=cross)
        self._logger.debug(f'Shape after = {self.shape}')

    @staticmethod
    def cut_phys_dim(*,
                     lnode: Node,
                     rnode: Node,
                     max_phys_dim: int = None,
                     **kwargs):
        assert (lnode.id is not None) and (rnode.id is not None)
        assert rnode.id in lnode.neighbours
        assert lnode.id in rnode.neighbours
        assert len(lnode.neighbours[rnode.id]) == 1
        assert len(rnode.neighbours[lnode.id]) == 1
        assert max_phys_dim is not None
        assert isinstance(lnode._tensor, MPS)
        assert isinstance(rnode._tensor, MPS)

        ldx = lnode._neighbours[rnode.id][0]
        rdx = rnode._neighbours[lnode.id][0]

        lnode._tensor.cut_phys_dim(ltensor=lnode._tensor,
                                   ldx=ldx,
                                   rtensor=rnode._tensor,
                                   rdx=rdx,
                                   max_phys_dim=max_phys_dim,
                                   **kwargs)

    @classmethod
    def contract(cls,
                 *,
                 lnode: Node,
                 rnode: Node,
                 **kwargs) -> Tuple[Node, tf.Tensor]:
        assert (lnode.id is not None) and (rnode.id is not None)
        assert rnode.id in lnode.neighbours
        assert lnode.id in rnode.neighbours
        assert len(lnode.neighbours[rnode.id]) == 1
        assert len(rnode.neighbours[lnode.id]) == 1

        logger = logging.getLogger(f'nnqs.{cls.__name__}')

        ldx = lnode._neighbours[rnode.id][0]
        rdx = rnode._neighbours[lnode.id][0]

        logger.info(f'Node #{lnode.id}, ldx = {ldx}\tNode #{rnode.id}, rdx = {rdx}')
        tensor, norm = lnode._tensor.contract_by_idx(ltensor=lnode._tensor,
                                                     ldx=ldx,
                                                     rtensor=rnode._tensor,
                                                     rdx=rdx,
                                                     **kwargs)
        neighbours = {}
        for nid in lnode._neighbours:
            if nid != rnode.id:
                neighbours[nid] = tuple(Node.gated_offset(x, gate=ldx) for x in lnode._neighbours[nid])
        offset = len(lnode.shape) - 1
        for nid in rnode._neighbours:
            if nid != lnode.id:
                if nid not in neighbours:
                    neighbours[nid] = tuple(offset + Node.gated_offset(x, gate=rdx)
                                            for x in rnode._neighbours[nid])
                else:
                    neighbours[nid] = neighbours[nid] + tuple(offset + Node.gated_offset(x, gate=rdx)
                                                              for x in rnode._neighbours[nid])
        lname = lnode._name if lnode._name is not None else f'{lnode._id}'
        rname = rnode._name if rnode._name is not None else f'{rnode._id}'
        name = f'({lname}_{ldx}-{rname}_{rdx})'

        return Node(name=name,
                    tensor=tensor,
                    neighbours=neighbours), norm

    @classmethod
    def calc_min_swap_num(cls,
                          *,
                          lnode: Node,
                          rnode: Node) -> int:
        return MPS.calc_min_swap_num(ltensor=lnode._tensor,
                                     lindices=lnode._neighbours[rnode._id],
                                     rtensor=rnode._tensor,
                                     rindices=rnode._neighbours[lnode._id])

    @classmethod
    def contract_by_indices(cls,
                            *,
                            lnode: Node,
                            rnode: Node,
                            **kwargs) -> Tuple[Node, tf.Tensor]:
        assert (lnode.id is not None) and (rnode.id is not None)
        assert rnode.id in lnode.neighbours
        assert lnode.id in rnode.neighbours
        assert len(lnode.neighbours[rnode.id]) == len(rnode.neighbours[lnode.id])
        assert isinstance(lnode._tensor, MPS)
        assert isinstance(rnode._tensor, MPS)

        logger = logging.getLogger(f'nnqs.{cls.__name__}')

        lindices = lnode._neighbours[rnode.id]
        rindices = rnode._neighbours[lnode.id]

        logger.info(f'Contracting by indices: ')
        logger.info(f'Node #{lnode.id}, lindices = {lindices}\tNode #{rnode.id}, rindices = {rindices}')
        tensor, norm, lold_to_cur, rold_to_cur = MPS.contract_by_indices(lmps=lnode._tensor,
                                                                         lindices=lindices,
                                                                         rmps=rnode._tensor,
                                                                         rindices=rindices,
                                                                         **kwargs)
        neighbours = {}
        for nid in lnode._neighbours:
            if nid != rnode.id:
                neighbours[nid] = tuple(lold_to_cur[idx] for idx in lnode._neighbours[nid])
        offset = lnode.rank - 2 * len(lindices)
        for nid in rnode._neighbours:
            if nid != lnode.id:
                if nid not in neighbours:
                    neighbours[nid] = tuple(offset + rold_to_cur[idx]
                                            for idx in rnode._neighbours[nid])
                else:
                    neighbours[nid] = neighbours[nid] + tuple(offset + rold_to_cur[idx]
                                                              for idx in rnode._neighbours[nid])
        lname = lnode._name if lnode._name is not None else f'{lnode._id}'
        rname = rnode._name if rnode._name is not None else f'{rnode._id}'
        name = f'({lname}_{lindices}-{rname}_{rindices})'

        return Node(name=name,
                    tensor=tensor,
                    neighbours=neighbours), norm
