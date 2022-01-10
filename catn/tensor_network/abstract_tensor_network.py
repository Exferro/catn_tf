import networkx as nx

import numpy as np

from abc import ABC
from abc import abstractmethod

from typing import Tuple

from ...logging import Logging
from ..node import Node


class AbstractTensorNetwork(Logging, ABC):
    def __init__(self,
                 name: str = None):
        super(AbstractTensorNetwork, self).__init__()
        self._name = name
        self._node_count = 0
        self._nodes = dict()
        self._name_to_id = dict()

    def add_node(self, node: Node = None):
        if node.name in self._name_to_id:
            raise RuntimeError(f'Trying to add duplicate Node {node.name} to the '
                               f'TensorNetwork {self._name}')
        node.id = self._node_count
        self._node_count += 1

        self._nodes[node.id] = node
        self._name_to_id[node.name] = node.id

    def delete_node(self, id: int = None):
        assert id is not None
        assert id in self._nodes
        assert self._nodes[id].name in self._name_to_id

        del self._name_to_id[self._nodes[id].name]
        del self._nodes[id]

    def connect_by_name(self,
                        lname: str = None,
                        ldx: int = None,
                        rname: str = None,
                        rdx: int = None):
        assert lname is not None
        assert ldx is not None
        assert rname is not None
        assert rdx is not None

        assert lname in self._name_to_id
        assert rname in self._name_to_id

        lid = self._name_to_id[lname]
        rid = self._name_to_id[rname]

        lnode = self._nodes[lid]
        rnode = self._nodes[rid]

        assert lnode.shape[ldx] == rnode.shape[rdx]
        lnode.add_neighbour(nid=rid, idx=ldx)
        rnode.add_neighbour(nid=lid, idx=rdx)

    def to_graph(self):
        graph = nx.Graph()
        for id, node in self._nodes.items():
            graph.add_node(id)
            for nid in node.neighbours:
                graph.add_edge(id, nid)

        return graph

    def size_after_contract(self, id, jd):
        inode = self._nodes[id]
        jnode = self._nodes[jd]
        assert jd in inode.neighbours
        assert id in jnode.neighbours

        ishape = np.asarray(inode.shape)
        jshape = np.asarray(jnode.shape)

        return (np.sum(np.log2(ishape))
                + np.sum(np.log2(jshape))
                - np.sum(np.log2(ishape[np.asarray(inode.neighbours[jd])]))
                - np.sum(np.log2(jshape[np.asarray(jnode.neighbours[id])])))

    def rearrange_neighbours(self, fnode, tnode):
        for nid in fnode.neighbours:
            if nid != tnode.id:
                if tnode.id in self._nodes[nid].neighbours:
                    self._nodes[nid].neighbours[tnode.id] = self._nodes[nid].neighbours[tnode.id] + \
                                                          self._nodes[nid].neighbours[fnode.id]

                else:
                    self._nodes[nid].neighbours[tnode.id] = self._nodes[nid].neighbours[fnode.id]
                del self._nodes[nid].neighbours[fnode.id]

    @abstractmethod
    def contract_edge(self, edge: Tuple[int, int] = None):
        ...

    @abstractmethod
    def contract(self):
        ...

    def get_shapes(self):
        node_ids = sorted(self._nodes.keys())
        result = {}
        for id in node_ids:
            result[id] = (self._nodes[id].shape, self._nodes[id].hidden_shape)

        return result

    def print_shape_changes(self, start_shapes, end_shapes):
        for start_id in start_shapes:
            if start_id not in end_shapes:
                self._logger.info(f'Node #{start_id}:\n'
                                  f'{start_shapes[start_id][0]} -> X\n'
                                  f'{start_shapes[start_id][1]} -> X')
            else:
                self._logger.info(f'Node #{start_id}:\n'
                                  f'{start_shapes[start_id][0]} -> {end_shapes[start_id][0]}\n'
                                  f'{start_shapes[start_id][1]} -> {end_shapes[start_id][1]}')
        end_id = list(end_shapes.keys())[-1]
        assert end_id not in start_shapes
        self._logger.info(f'Node #{end_id}:\n'
                          f' -> {end_shapes[end_id][0]}\n'
                          f' -> {end_shapes[end_id][1]}')
