from __future__ import annotations

import time

from enum import Enum
from typing import Tuple

from . import AbstractTensorNetwork
from .. import Node

EdgeSelection = Enum('EdgeSelection', 'dummy min_size')


class BaseTensorNetwork(AbstractTensorNetwork):
    def __init__(self,
                 *,
                 max_phys_dim: int = None,
                 minimise_swaps: bool = False,
                 edge_selection: EdgeSelection = EdgeSelection.min_size,
                 **kwargs):
        super(BaseTensorNetwork, self).__init__()
        self._max_phys_dim = max_phys_dim
        self._edge_selection = edge_selection

        # TODO: remove this field once its influence is clear
        self._minimise_swaps = minimise_swaps

    def select_edge_dummy(self, edges):

        return edges[0]

    def select_edge_min_size(self, edges):
        sizes_after_contraction = [self.size_after_contract(*edge) for edge in edges]
        sizes_after_contraction = sorted(enumerate(sizes_after_contraction), key=lambda tup: tup[1])

        return edges[sizes_after_contraction[0][0]]

    def select_edge(self, edge_selection: EdgeSelection, edges):
        if edge_selection == EdgeSelection.dummy:
            return self.select_edge_dummy(edges)
        elif edge_selection == EdgeSelection.min_size:
            return self.select_edge_min_size(edges)
        else:
            raise ValueError(f'Unknown edge selection algorithm: {edge_selection}')

    def merge_duplicate(self, node: Node = None, jdx_neighbours: list = None):
        """
        Goes through all neighbours of a given node and merges all duplicate connections
        (performing merge for physical indices of neighbours too).

        :param node:
        :return:
        """
        duplicate_nodes = []
        start_time = time.time()
        for nid in jdx_neighbours:
            assert (len(node.neighbours[nid]) == 2) or (len(node.neighbours[nid]) == 1)
            if len(node.neighbours[nid]) == 2:
                assert len(self._nodes[nid].neighbours[node.id]) == 2
                duplicate_nodes.append(nid)
                new_idx_pair = node.neighbours[nid]
                old_idx_pair = self._nodes[nid].neighbours[node.id]
                cross = ((new_idx_pair[0] < new_idx_pair[1])
                         != (old_idx_pair[0] < old_idx_pair[1]))
                self._nodes[nid].merge_duplicate(node.id, cross=cross)
        print(f'Time spent on merging indices in {len(duplicate_nodes)} neighbours = {time.time() - start_time}')

        neigh_time = 0.0
        cut_time = 0.0
        self._logger.debug(f'New Node #{node.id}, shape = {node.shape}')
        neigh_idx = 1
        for nid in duplicate_nodes:
            start_time = time.time()
            node.merge_duplicate(nid, cross=False)
            neigh_time = time.time() - start_time
            print(f'Time spent on moving neighbour #{neigh_idx}: {neigh_time}')
            neigh_idx += 1

            if self._max_phys_dim is not None:
                if node.shape[node.neighbours[nid][0]] > self._max_phys_dim:
                    start_time = time.time()
                    Node.cut_phys_dim(lnode=node,
                                      rnode=self._nodes[nid],
                                      max_phys_dim=self._max_phys_dim)
                    cut_time += time.time() - start_time
        #print(f'Time spent on merging indices in the node = {neigh_time}')
        print(f'Time spent on cutting the dimension = {cut_time}')

    def minimise_swaps(self,
                       lid: int,
                       rid: int) -> Tuple[int, int]:
        lswaps, rswaps = (self._nodes[lid].rank - 1) - lid, rid
        if (lswaps + rswaps) > (self._nodes[lid].rank - 1 - lswaps) + (self._nodes[rid]._tensor.rank - 1 - rswaps):
            lid, rid = rid, lid

        return lid, rid

    def contract_edge(self, edge: Tuple[int, int] = None):
        if self._minimise_swaps:
            id, jd = self.minimise_swaps(*edge)
        else:
            id, jd = edge[0], edge[1]

        sorted_jdx = sorted([(id, idx[0]) for id, idx in self._nodes[jd]._neighbours.items()],
                            key=lambda tup: tup[1])
        jdx_neighbours = [tup[0] for tup in sorted_jdx if tup[0] != id]
        new_node, norm = Node.contract(lnode=self._nodes[id],
                                       rnode=self._nodes[jd])
        self.add_node(new_node)

        self.rearrange_neighbours(self._nodes[id], new_node)
        self.rearrange_neighbours(self._nodes[jd], new_node)

        self.delete_node(id)
        self.delete_node(jd)

        self.merge_duplicate(new_node, jdx_neighbours=jdx_neighbours)
        self._logger.debug(f'Shape of new node after merge duplicate: {new_node.shape}')
        self._logger.debug(f'Number of edges left: {len(self.to_graph().edges)}\n')

        return norm

    def contract(self):
        edges = list(self.to_graph().edges)
        iter_idx = 1
        result = 1.0
        self._logger.info(f'Contraction of TensorNetwork {self._name} started')
        iter_to_do = len(self._nodes) - 1
        max_iter_time = 0.0
        max_iter_idx = -1
        while len(edges):
            self._logger.info(f'Iteration {iter_idx}/{iter_to_do}')

            start_shapes = self.get_shapes()
            start_time = time.time()
            norm = self.contract_edge(self.select_edge(self._edge_selection, edges))
            iter_time = time.time() - start_time
            if iter_time > max_iter_time:
                max_iter_time = iter_time
                max_iter_idx = iter_idx
            result *= norm
            end_shapes = self.get_shapes()

            edges = list(self.to_graph().edges)
            self._logger.info('')
            iter_idx += 1

        assert len(self._nodes) == 1
        self._logger.info(f'Finished')
        #print(f'Max iter idx = {max_iter_idx}, time = {max_iter_time}')

        return self._nodes[self._node_count - 1].part_func() * result
