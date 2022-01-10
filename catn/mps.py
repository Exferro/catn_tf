from __future__ import annotations

import logging
import time

import numpy as np
import scipy
import pandas as pd

import tensorflow as tf
import torch
from typing import Tuple, Union, Dict

from scipy.sparse import linalg as spslinalg
from scipy.sparse import csr_matrix

from ..constants import BASE_INT_TYPE
from ..constants import BASE_COMPLEX_TYPE
from ..constants import INT_TYPE_TO_BIT_DEPTH

from ..logging import Logging
from ..picklable import Picklable
from .abstract_tensor import AbstractTensor

from ..svd import trunc_svd
from .constants import DEFAULT_CUTOFF, DEFAULT_MAX_PHYS_DIM, DEFAULT_MAX_BOND_DIM
from .constants import BACKENDS, DEFAULT_SVD_BACKEND, DEFAULT_BACKPROP_BACKEND
from .constants import CANO_DECOMPS, DEFAULT_CANO_DECOMP
from .constants import DEFAULT_MPS_OPERATION_MODE

from .projector import Projector

from ..tensor_initialiser import TensorInitialiser
from ..custom_diag import custom_diag


class MPS(AbstractTensor, Picklable, Logging):
    SVD_BACKEND = DEFAULT_SVD_BACKEND
    CANO_DECOMP = DEFAULT_CANO_DECOMP
    OPERATION_MODE = DEFAULT_MPS_OPERATION_MODE

    SVD_MATRICES_ROOT = './'
    SVD_MATRICES_NUM = 0

    SVD_STAT_COLS = ('ext_contractor',
                     'backend',
                     'type',
                     'height',
                     'width',
                     'svd_time',
                     'init_bond_dim',
                     'cutoff_dim',
                     'max_bond_dim',
                     'new_bond_dim',
                     'matrix_id',
                     'sparseness')
    SVD_STAT = pd.DataFrame(columns=SVD_STAT_COLS)

    EXT_CONTRACTOR = None

    def __init__(self,
                 *,
                 name: str = None,
                 visible_num: int,
                 phys_dims: Union[int, list] = None,
                 bond_dims: Union[int, list] = None,
                 given_orth_idx: int = None,
                 new_orth_idx: int = None,
                 max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                 cutoff: float = DEFAULT_CUTOFF,
                 dtype=BASE_COMPLEX_TYPE,
                 idx_dtype=BASE_INT_TYPE,
                 tensors: list = None,
                 init_method=None):
        super(MPS, self).__init__(name=name,
                                  dtype=dtype)
        self._idx_dtype = idx_dtype

        self._visible_num = visible_num
        if self._visible_num > INT_TYPE_TO_BIT_DEPTH[self._idx_dtype]:
            self._logger.warning(f'Number of physical indices in the MPS {self._name} '
                                 f'{self._visible_num} is larger than bit depth of '
                                 f'underlying integer data type '
                                 f'{self._idx_dtype}: {INT_TYPE_TO_BIT_DEPTH[self._idx_dtype]}. '
                                 f'Please, be careful calling amplitude() member function.')

        if isinstance(phys_dims, int):
            self._phys_dims = [phys_dims] * visible_num
        elif isinstance(phys_dims, list):
            assert len(phys_dims) == visible_num
            self._phys_dims = [phys_dim for phys_dim in phys_dims]
        else:
            raise TypeError(f'phys_dims should be either int, or list. '
                            f'In fact they are: {type(bond_dims)}.')

        if isinstance(bond_dims, int):
            self._bond_dims = [bond_dims] * (visible_num - 1)
        elif isinstance(bond_dims, list):
            if visible_num > 0:
                assert len(bond_dims) == (visible_num - 1)
            self._bond_dims = [bond_dim for bond_dim in bond_dims]
        else:
            raise TypeError(f'bond_dims should be either int, or list. '
                            f'In fact they are: {type(bond_dims)}.')

        self._ext_bond_dims = [1] + [bond_dim for bond_dim in self._bond_dims] + [1]

        self._max_bond_dim = max_bond_dim
        self._cutoff = cutoff

        if tensors is None:
            tensor_initialiser = TensorInitialiser(dtype=self._dtype, init_method=init_method)

            # Initialise tensors
            self._tensors = [tensor_initialiser((self._ext_bond_dims[idx],
                                                 self._phys_dims[idx],
                                                 self._ext_bond_dims[idx + 1]))
                             for idx in range(self._visible_num)]
        else:
            assert np.all([tensor.dtype == self._dtype for tensor in tensors])
            assert init_method is None
            assert isinstance(tensors, list)
            if self._visible_num > 0:
                assert (len(tensors) == self._visible_num)

                # Check all tensors are 3-way
                assert np.all([len(tensor.shape) == 3 for tensor in tensors])

                # Check consistency of all physical and bond dimensions
                input_phys_dims = [tensor.shape[1] for tensor in tensors]
                assert np.all(np.equal(input_phys_dims, self._phys_dims))

                input_bond_dims = [tensor.shape[2] for tensor in tensors[:-1]]
                assert np.all(np.equal(input_bond_dims, self._bond_dims))

                input_bond_dims = [tensor.shape[0] for tensor in tensors[1:]]
                assert np.all(np.equal(input_bond_dims, self._bond_dims))

            # Check left and right caps are caps
            assert tensors[0].shape[0] == 1
            assert tensors[-1].shape[-1] == 1

            self._tensors = tensors

        assert MPS.SVD_BACKEND in BACKENDS
        assert MPS.CANO_DECOMP in CANO_DECOMPS

        self._orth_idx = given_orth_idx
        if new_orth_idx is not None:
            if self._visible_num > 0:
                self._canonicalise(new_orth_idx)

        # A dict which keeps track of all index permutations (which effectively
        # happen only during the swap_adjacent execution)
        self._cur_to_old = {idx: idx for idx in range(visible_num)}

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._phys_dims)

    @property
    def hidden_shape(self) -> Tuple[int, ...]:
        return tuple(self._ext_bond_dims)

    def __str__(self) -> str:

        return (f'MPS {self._name}:\n'
                f'\tvisible_num = {self._visible_num}\n'
                f'\tphys_dims = {self._phys_dims}\n'
                f'\tbond_dims = {self._bond_dims}\n'
                f'\text_bond_dims = {self._ext_bond_dims}\n'
                f'\torth_idx = {self._orth_idx}\n')

    def _idx_to_visible(self, idx):
        return tf.cast(tf.squeeze(tf.math.floormod(tf.bitwise.right_shift(tf.reshape(idx, (-1, 1)),
                                                                          tf.range(self._visible_num,
                                                                                   dtype=self._idx_dtype)),
                                                   2)),
                       dtype=self._idx_dtype)

    def _visible_to_idx(self, visible):
        if visible.dtype != self._idx_dtype:
            visible = tf.cast(visible, self._idx_dtype)
        two_powers = 2 ** tf.range(0, self._visible_num, dtype=self._idx_dtype)

        return tf.tensordot(visible, two_powers, axes=1)

    def _visible_to_amplitude(self, visible):
        """
        Batched calculation of MPS amplitudes corresponding to the ndarray
        of input visible states with shape (batch_index, visible_num).

        :param visible:
        :return:
        """
        assert visible.shape[-1] == self._visible_num
        rolling_tensor = tf.gather(self._tensors[0], visible[:, 0], axis=1)
        for idx in range(1, self._visible_num):
            rolling_tensor = tf.einsum('iaj,jak->iak',
                                       rolling_tensor,
                                       tf.gather(self._tensors[idx], visible[:, idx], axis=1))

        return tf.squeeze(rolling_tensor)

    def amplitude(self, idx: tf.Tensor) -> tf.Tensor:

        return self._visible_to_amplitude(self._idx_to_visible(idx))

    def part_func(self):
        result = tf.reduce_sum(self._tensors[0], axis=1)
        for idx in range(1, self._visible_num):
            result = tf.matmul(result, tf.reduce_sum(self._tensors[idx], axis=1))

        return tf.reduce_sum(result)

    def norm(self):
        # Up -> bottom
        result = tf.squeeze(tf.tensordot(self._tensors[0],
                                         tf.math.conj(self._tensors[0]),
                                         axes=[1, 1]))
        for idx in range(1, self._visible_num):
            # Left -> right
            result = tf.tensordot(result,
                                  tf.math.conj(self._tensors[idx]),
                                  axes=[-1, 0])

            # Up -> bottom
            result = tf.tensordot(self._tensors[idx],
                                  result,
                                  axes=[[0, 1], [0, 1]])

        return tf.squeeze(tf.sqrt(result))

    def to_tensor(self):
        result = self._tensors[0]
        for idx in range(1, self._visible_num):
            result = tf.tensordot(result, self._tensors[idx], axes=[-1, 0])

        return tf.squeeze(result, axis=[0, -1])

    def to_state_vector(self):
        return self.amplitude(tf.range(2 ** self._visible_num))

    @staticmethod
    def truncated_svd(*,
                      matrix: tf.Tensor = None,
                      cutoff: float = DEFAULT_CUTOFF,
                      max_bond_dim: int = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        assert len(matrix.shape) == 2
        assert MPS.SVD_BACKEND in BACKENDS

        was_sparse = False

        start_time = time.time()
        if MPS.SVD_BACKEND == 'TF':
            s, u, v = tf.linalg.svd(matrix)
        elif MPS.SVD_BACKEND == 'TORCH':
            u, s, v = torch.svd(torch.from_numpy(matrix.numpy()))
            u = tf.constant(u.numpy(), dtype=matrix.dtype)
            s = tf.constant(s.numpy(), dtype=matrix.dtype)
            v = tf.constant(v.numpy(), dtype=matrix.dtype)
        elif MPS.SVD_BACKEND == 'SCIPY':
            if MPS.OPERATION_MODE == 'FAST':
                if min(matrix.shape[0], matrix.shape[1]) > max_bond_dim:
                    was_sparse = True
                    sparseness = np.isclose(matrix, np.zeros_like(matrix)).sum() / (matrix.shape[0] * matrix.shape[1])
                    if sparseness >= 0.95:
                        sparse_matrix = csr_matrix(np.where(np.isclose(matrix,
                                                                       np.zeros_like(matrix),
                                                                       atol=1e-15),
                                                            np.zeros_like(matrix.numpy()),
                                                            matrix.numpy()))
                    else:
                        sparse_matrix = matrix.numpy()
                    sparse_matrix = matrix.numpy()
                    u, s, v_h = spslinalg.svds(sparse_matrix, k=max_bond_dim)
                    u = u[:, ::-1]
                    s = s[::-1]
                    v_h = v_h[::-1, :]
                else:
                    u, s, v_h = scipy.linalg.svd(matrix.numpy(), full_matrices=False)
            elif MPS.OPERATION_MODE == 'SLOW':
                u, s, v_h = scipy.linalg.svd(matrix.numpy(), full_matrices=False)
            else:
                raise ValueError(f'Wrong MPS operation mode: {MPS.OPERATION_MODE}')

            u = tf.constant(u, dtype=matrix.dtype)
            s = tf.constant(s, dtype=matrix.dtype)
            v = tf.linalg.adjoint(tf.constant(v_h, dtype=matrix.dtype))

            argsort = tf.argsort(tf.math.real(s), direction='DESCENDING')
            u = tf.gather(u, argsort, axis=1)
            s = tf.gather(s, argsort)
            v = tf.gather(v, argsort, axis=1)
        else:
            raise ValueError(f'Somehow wrong backend {MPS.SVD_BACKEND} leaked through the assert '
                             f'in MPS.truncated_svd')
        svd_time = time.time() - start_time

        # Truncate singular values which are too small
        init_bond_dim = s.shape[0]
        cutoff_dim = len(s.numpy()[s.numpy() > cutoff])
        new_bond_dim = min(max_bond_dim, cutoff_dim) if max_bond_dim is not None else cutoff_dim
        if new_bond_dim == 0:
            logger = logging.getLogger(f'nnqs.MPS')
            logger.warning(f'Zero new_bond_dim encountered during truncated_svd')
            new_bond_dim = 1
        s = tf.cast(tf.linalg.diag(s[:new_bond_dim]), matrix.dtype)
        u = u[:, :new_bond_dim]
        v_t = tf.linalg.adjoint(v[:, :new_bond_dim])

        #np.save(os.path.join(MPS.SVD_MATRICES_ROOT, f'{MPS.SVD_MATRICES_NUM}'),
        #        matrix.numpy())
        summary = {
            'ext_contractor': MPS.EXT_CONTRACTOR,
            'backend': MPS.SVD_BACKEND,
            'type': 'sparse' if was_sparse else 'dense',
            'height': matrix.shape[0],
            'width': matrix.shape[1],
            'svd_time': svd_time,
            'init_bond_dim': init_bond_dim,
            'cutoff_dim': cutoff_dim,
            'max_bond_dim': max_bond_dim,
            'new_bond_dim': new_bond_dim,
            'matrix_id': MPS.SVD_MATRICES_NUM,
            'sparseness': np.isclose(matrix, np.zeros_like(matrix), atol=1e-15).sum() / (matrix.shape[0] * matrix.shape[1])
        }
        #MPS.SVD_STAT = MPS.SVD_STAT.append([summary], ignore_index=True)
        MPS.SVD_MATRICES_NUM += 1

        return u, s, v_t

    @staticmethod
    def from_tensor(*,
                    name: str = None,
                    tensor: tf.Tensor = None,
                    new_orth_idx: int = None,
                    max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                    cutoff: float = DEFAULT_CUTOFF,
                    idx_dtype=BASE_INT_TYPE) -> MPS:
        phys_dims = list(tensor.shape)
        visible_num = len(phys_dims)
        tensors = []
        bond_dims = []
        ext_bond_dims = [1]
        for idx in range(visible_num - 1):
            tensor = tf.reshape(tensor, (ext_bond_dims[idx] * phys_dims[idx], -1))

            u, s, v_t = MPS.truncated_svd(matrix=tensor,
                                          cutoff=cutoff,
                                          max_bond_dim=max_bond_dim)
            bond_dims.append(u.shape[1])
            ext_bond_dims.append(u.shape[1])

            tensors.append(tf.reshape(u, (ext_bond_dims[idx], phys_dims[idx], ext_bond_dims[idx + 1])))
            tensor = tf.matmul(s, v_t)

        ext_bond_dims.append(1)
        tensors.append(tf.reshape(tensor, (ext_bond_dims[visible_num - 1],
                                           phys_dims[visible_num - 1],
                                           ext_bond_dims[visible_num])))

        return MPS(name=name,
                   visible_num=visible_num,
                   phys_dims=phys_dims,
                   bond_dims=bond_dims,
                   given_orth_idx=visible_num - 1,
                   new_orth_idx=new_orth_idx,
                   max_bond_dim=max_bond_dim,
                   cutoff=cutoff,
                   dtype=tensor.dtype,
                   idx_dtype=idx_dtype,
                   tensors=tensors)

    def _set_bond_dim(self, bond_idx, val):
        """
        A function to simultaneously update an entry in the list of bond
        dimensions (`self._bond_dims`) and in the extended list of bond
        dimensions (`self._ext_bond_dims`)

        :param bond_idx:
        :param val:
        :return:
        """
        self._bond_dims[bond_idx] = val
        self._ext_bond_dims[bond_idx + 1] = val

    def _canonicalise(self, new_orth_idx):
        assert (0 <= new_orth_idx) and (new_orth_idx < self._visible_num)

        if self._orth_idx is None:
            forward_start_idx = 0
            backward_start_idx = self._visible_num - 1
        else:
            forward_start_idx = self._orth_idx
            backward_start_idx = self._orth_idx

        for idx in range(forward_start_idx, new_orth_idx):
            matrix = tf.reshape(self._tensors[idx],
                                (self._ext_bond_dims[idx] * self._phys_dims[idx],
                                 self._ext_bond_dims[idx + 1]))
            if MPS.CANO_DECOMP == 'QR':
                q, r = tf.linalg.qr(matrix)
            else:
                u, s, v = self.truncated_svd(matrix=matrix,
                                             cutoff=self._cutoff,
                                             max_bond_dim=self._max_bond_dim)
                q, r = u, s @ v

            self._set_bond_dim(idx, q.shape[1])
            self._tensors[idx] = tf.reshape(q, (self._ext_bond_dims[idx],
                                                self._phys_dims[idx],
                                                self._ext_bond_dims[idx + 1]))
            self._tensors[idx + 1] = tf.tensordot(r,
                                                  self._tensors[idx + 1],
                                                  axes=[1, 0])

        for idx in range(backward_start_idx, new_orth_idx, -1):
            matrix = tf.transpose(tf.reshape(self._tensors[idx],
                                             (self._ext_bond_dims[idx],
                                              self._ext_bond_dims[idx + 1] * self._phys_dims[idx])))
            if MPS.CANO_DECOMP == 'QR':
                q_t, r_t = tf.linalg.qr(matrix)
            else:
                u, s, v = self.truncated_svd(matrix=matrix,
                                             cutoff=self._cutoff,
                                             max_bond_dim=self._max_bond_dim)
                q_t, r_t = u, s @ v
            q = tf.transpose(q_t)
            r = tf.transpose(r_t)

            self._set_bond_dim(idx - 1, q.shape[0])
            self._tensors[idx] = tf.reshape(q, (self._ext_bond_dims[idx],
                                                self._phys_dims[idx],
                                                self._ext_bond_dims[idx + 1]))
            self._tensors[idx - 1] = tf.tensordot(self._tensors[idx - 1],
                                                  r,
                                                  axes=[-1, 0])

        self._orth_idx = new_orth_idx

    def cut_bond_dims(self,
                      *,
                      svd_backend: str = DEFAULT_SVD_BACKEND,
                      backprop_backend: str = DEFAULT_BACKPROP_BACKEND):
        for bond_idx, bond_dim in enumerate(self._bond_dims):
            if bond_dim > self._max_bond_dim:
                ldx, rdx = bond_idx, bond_idx + 1
                self._canonicalise(ldx)

                bond_tensor = tf.einsum('iaj,jbk->iabk',
                                        self._tensors[ldx],
                                        self._tensors[rdx])
                # Calculate external bond dimensions of left and right matrices
                left_dim = self._ext_bond_dims[ldx] * self._phys_dims[ldx]
                right_dim = self._ext_bond_dims[rdx + 1] * self._phys_dims[rdx]

                bond_tensor = tf.reshape(bond_tensor, (left_dim, right_dim))
                u, s, v = trunc_svd(matrix=bond_tensor,
                                    max_bond_dim=self._max_bond_dim,
                                    cutoff=self._cutoff,
                                    svd_backend=svd_backend,
                                    backprop_backend=backprop_backend)
                self._set_bond_dim(ldx, u.shape[1])
                ltensor = tf.matmul(u, tf.linalg.diag(s))
                rtensor = tf.linalg.adjoint(v)

                self._tensors[ldx] = tf.reshape(ltensor, (self._ext_bond_dims[ldx],
                                                          self._phys_dims[ldx],
                                                          self._ext_bond_dims[ldx + 1]))
                self._tensors[rdx] = tf.reshape(rtensor, (self._ext_bond_dims[rdx],
                                                          self._phys_dims[rdx],
                                                          self._ext_bond_dims[rdx + 1]))

    def swap_adjacent(self, fdx, tdx):
        """
        Swaps neighbouring physical indices fdx and tdx.
        Places the orthogonality center in the tdx-th tensor.

        :param fdx:
        :param tdx:
        :return:
        """
        assert (0 <= fdx) and (fdx < self._visible_num)
        assert (0 <= tdx) and (tdx < self._visible_num)

        assert abs(fdx - tdx) == 1

        (ldx, rdx) = (min(fdx, tdx), max(fdx, tdx))
        if self._orth_idx is None:
            self._canonicalise(ldx)
        elif self._orth_idx < ldx:
            self._canonicalise(ldx)
        elif self._orth_idx > rdx:
            self._canonicalise(rdx)

        # Calculate merged tensor and swap its physical axes
        contracted_tensor = tf.einsum('iaj,jbk->ibak',
                                      self._tensors[ldx],
                                      self._tensors[rdx])

        # Calculate external bond dimensions of left and right matrices
        left_dim = self._ext_bond_dims[ldx] * self._phys_dims[rdx]
        right_dim = self._ext_bond_dims[rdx + 1] * self._phys_dims[ldx]

        contracted_tensor = tf.reshape(contracted_tensor,
                                       (left_dim, right_dim))
        u, s, v_t = self.truncated_svd(matrix=contracted_tensor,
                                       cutoff=self._cutoff,
                                       max_bond_dim=self._max_bond_dim)
        self._set_bond_dim(ldx, u.shape[1])

        if fdx < tdx:
            ltensor = u
            rtensor = tf.matmul(s, v_t)
        else:
            ltensor = tf.matmul(u, s)
            rtensor = v_t
        self._orth_idx = tdx

        self._tensors[ldx] = tf.reshape(ltensor, (self._ext_bond_dims[ldx],
                                                  self._phys_dims[rdx],
                                                  self._ext_bond_dims[ldx + 1]))
        self._tensors[rdx] = tf.reshape(rtensor, (self._ext_bond_dims[rdx],
                                                  self._phys_dims[ldx],
                                                  self._ext_bond_dims[rdx + 1]))
        self._phys_dims[ldx] = self._tensors[ldx].shape[1]
        self._phys_dims[rdx] = self._tensors[rdx].shape[1]

        self._cur_to_old[fdx], self._cur_to_old[tdx] = self._cur_to_old[tdx], self._cur_to_old[fdx]

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
            self._canonicalise(tdx)

    def old_to_cur(self):

        return {self._cur_to_old[cur_idx]: cur_idx for cur_idx in self._cur_to_old}

    def move_to_tail(self, fdx):
        if fdx < 0:
            raise ValueError(f"move_to_tail(): fdx should be larger than 0, fdx = {fdx}")
        if fdx == self._visible_num - 1:
            self._canonicalise(new_orth_idx=self._visible_num - 1)
            return
        for idx in range(fdx, self._visible_num - 1):
            self.swap_adjacent(idx, idx + 1)
        self._canonicalise(new_orth_idx=self._visible_num - 1)

    def merge_adjacent(self, idx, jdx):
        assert (0 <= idx) and (idx < self.rank)
        assert (0 <= jdx) and (jdx < self.rank)

        assert abs(idx - jdx) == 1

        ldx, rdx = min(idx, jdx), max(idx, jdx)
        #self._canonicalise(ldx)
        tensor = tf.tensordot(self._tensors[ldx],
                              self._tensors[rdx],
                              axes=[-1, 0])
        tensor = tf.reshape(tensor, (self._ext_bond_dims[ldx],
                                     self._phys_dims[ldx] * self._phys_dims[rdx],
                                     self._ext_bond_dims[rdx + 1]))
        self._visible_num -= 1

        self._phys_dims[ldx] = self._phys_dims[ldx] * self._phys_dims[rdx]
        self._phys_dims.pop(rdx)

        self._bond_dims.pop(ldx)
        self._ext_bond_dims.pop(rdx)

        self._tensors.pop(rdx)
        self._tensors[ldx] = tensor

        #self._canonicalise(ldx)
        if self._orth_idx is not None:
            self._orth_idx = self._orth_idx - 1 if self._orth_idx > ldx else self._orth_idx

    @staticmethod
    def cut_phys_dim_slow(*,
                          ltensor: MPS,
                          ldx: int,
                          rtensor: MPS,
                          rdx: int,
                          max_phys_dim: int = DEFAULT_MAX_PHYS_DIM,
                          **kwargs):
        assert ltensor._dtype == rtensor._dtype
        assert ltensor._idx_dtype == rtensor._idx_dtype
        assert ltensor._max_bond_dim == rtensor._max_bond_dim
        assert ltensor._cutoff == rtensor._cutoff

        assert ltensor._phys_dims[ldx] == rtensor._phys_dims[rdx]

        assert max_phys_dim is not None
        ltensor._canonicalise(ldx)
        rtensor._canonicalise(rdx)

        contracted_tensor = tf.einsum(f'iaj,kal->ijkl',
                                      ltensor._tensors[ldx],
                                      rtensor._tensors[rdx])
        contracted_tensor = tf.reshape(contracted_tensor, (ltensor._ext_bond_dims[ldx]
                                                           * ltensor._ext_bond_dims[ldx + 1],
                                                           rtensor._ext_bond_dims[rdx]
                                                           * rtensor._ext_bond_dims[rdx + 1]))
        u, s, v_t = MPS.truncated_svd(matrix=contracted_tensor,
                                      cutoff=ltensor._cutoff,
                                      max_bond_dim=max_phys_dim)
        sqrt_s = tf.sqrt(s)
        new_phys_dim = s.shape[1]
        lmatrix = u @ sqrt_s
        rmatrix = sqrt_s @ v_t

        ltensor._tensors[ldx] = tf.transpose(tf.reshape(lmatrix, (ltensor._ext_bond_dims[ldx],
                                                                  ltensor._ext_bond_dims[ldx + 1],
                                                                  new_phys_dim)),
                                             perm=(0, 2, 1))
        ltensor._phys_dims[ldx] = new_phys_dim

        rtensor._tensors[rdx] = tf.transpose(tf.reshape(rmatrix, (new_phys_dim,
                                                                  rtensor._ext_bond_dims[rdx],
                                                                  rtensor._ext_bond_dims[rdx + 1])),
                                             perm=(1, 0, 2))
        rtensor._phys_dims[rdx] = new_phys_dim

    @staticmethod
    def cut_phys_dim_fast(*,
                          ltensor: MPS,
                          ldx: int,
                          rtensor: MPS,
                          rdx: int,
                          max_phys_dim: int = DEFAULT_MAX_PHYS_DIM,
                          **kwargs):
        assert ltensor._dtype == rtensor._dtype
        assert ltensor._idx_dtype == rtensor._idx_dtype
        assert ltensor._max_bond_dim == rtensor._max_bond_dim
        assert ltensor._cutoff == rtensor._cutoff

        assert ltensor._phys_dims[ldx] == rtensor._phys_dims[rdx]

        assert max_phys_dim is not None
        ltensor._canonicalise(ldx)
        rtensor._canonicalise(rdx)

        lfull_matrix = tf.reshape(tf.transpose(ltensor._tensors[ldx],
                                               perm=(0, 2, 1)),
                                  (ltensor._ext_bond_dims[ldx] * ltensor._ext_bond_dims[ldx + 1],
                                   ltensor._phys_dims[ldx]))
        rfull_matrix = tf.reshape(tf.transpose(rtensor._tensors[rdx],
                                               perm=(1, 0, 2)),
                                  (rtensor._phys_dims[rdx],
                                   rtensor._ext_bond_dims[rdx] * rtensor._ext_bond_dims[rdx + 1]))

        projector = Projector(a_full=lfull_matrix, b_full=rfull_matrix)
        ltrunc_matrix, rtrunc_matrix = projector.project(max_bond_dim=max_phys_dim,
                                                         tol=1e-5,
                                                         cutoff=ltensor._cutoff)
        new_phys_dim = ltrunc_matrix.shape[1]
        ltensor._tensors[ldx] = tf.transpose(tf.reshape(ltrunc_matrix, (ltensor._ext_bond_dims[ldx],
                                                                        ltensor._ext_bond_dims[ldx + 1],
                                                                        new_phys_dim)),
                                             perm=(0, 2, 1))
        ltensor._phys_dims[ldx] = new_phys_dim

        rtensor._tensors[rdx] = tf.transpose(tf.reshape(rtrunc_matrix, (new_phys_dim,
                                                                        rtensor._ext_bond_dims[rdx],
                                                                        rtensor._ext_bond_dims[rdx + 1])),
                                             perm=(1, 0, 2))
        rtensor._phys_dims[rdx] = new_phys_dim

    @staticmethod
    def cut_phys_dim(*,
                     ltensor: MPS,
                     ldx: int,
                     rtensor: MPS,
                     rdx: int,
                     max_phys_dim: int = DEFAULT_MAX_PHYS_DIM,
                     **kwargs):
        if MPS.OPERATION_MODE == 'FAST':
            MPS.cut_phys_dim_fast(ltensor=ltensor,
                                  ldx=ldx,
                                  rtensor=rtensor,
                                  rdx=rdx,
                                  max_phys_dim=max_phys_dim)
        elif MPS.OPERATION_MODE == 'SLOW':
            MPS.cut_phys_dim_slow(ltensor=ltensor,
                                  ldx=ldx,
                                  rtensor=rtensor,
                                  rdx=rdx,
                                  max_phys_dim=max_phys_dim)
        else:
            raise ValueError(f'Wrong MPS operation mode: {MPS.OPERATION_MODE}')

    @staticmethod
    def contract_by_idx(ltensor: MPS,
                        ldx: int,
                        rtensor: MPS,
                        rdx: int,
                        name: str = None,
                        new_orth_idx: int = None) -> Tuple[MPS, tf.Tensor]:
        """
        Contracts two MPS by one physical index and returns the MPS representation of the
        resulting tensor (all calculations are performed in MPS representations, no full tensors
        are obtained at any point). If orth_idx is not specified (is None), sets orthonogality
        center position to `lmps._visible_num - 2`.

        :param ltensor:
        :param ldx:
        :param rtensor:
        :param rdx:
        :param name:
        :param new_orth_idx:
        :return:
        """
        assert ltensor._dtype == rtensor._dtype
        assert ltensor._idx_dtype == rtensor._idx_dtype
        assert ltensor._max_bond_dim == rtensor._max_bond_dim
        assert ltensor._cutoff == rtensor._cutoff

        assert ltensor._phys_dims[ldx] == rtensor._phys_dims[rdx]
        # Align MPSes so that left and right caps are contracted
        ltensor.move(ldx, ltensor.rank - 1)
        rtensor.move(rdx, 0)

        bond_tensor = tf.matmul(tf.reshape(ltensor._tensors[-1],
                                           (ltensor._ext_bond_dims[-2],
                                            ltensor._phys_dims[-1])),
                                tf.reshape(rtensor._tensors[0],
                                           (rtensor._phys_dims[0],
                                            rtensor._ext_bond_dims[1])))

        tensors = ltensor._tensors[:-1] + rtensor._tensors[1:]
        if ltensor.rank > 1:
            tensors[ltensor.rank - 2] = tf.einsum('iaj,jk->iak',
                                                  tensors[ltensor.rank - 2],
                                                  bond_tensor)

            bond_dims = ltensor._bond_dims[:-1] + rtensor._bond_dims
        elif rtensor.rank > 1:
            tensors[0] = tf.einsum('ij,jak->iak',
                                   bond_tensor,
                                   tensors[0])
            bond_dims = rtensor._bond_dims[1:]
        else:
            assert bond_tensor.shape == (1, 1)
            tensors.append(bond_tensor)
            bond_dims = []

        norm = None
        if ltensor.rank > 1:
            norm = tf.norm(tensors[ltensor.rank - 2])
            tensors[ltensor.rank - 2] = tf.divide(tensors[ltensor.rank - 2], norm)
        else:
            norm = tf.norm(tensors[0])
            tensors[0] = tf.divide(tensors[0], norm)

        name = f'({ltensor._name}_{ldx}-{rtensor._name}_{rdx})' if name is None else name

        visible_num = (ltensor.rank - 1) + (rtensor.rank - 1)
        phys_dims = ltensor._phys_dims[:-1] + rtensor._phys_dims[1:]

        return MPS(name=name,
                   visible_num=visible_num,
                   phys_dims=phys_dims,
                   bond_dims=bond_dims,
                   given_orth_idx=ltensor.rank - 2 if ltensor.rank > 1 else 0,
                   new_orth_idx=new_orth_idx if new_orth_idx is not None else visible_num - 1,
                   max_bond_dim=ltensor._max_bond_dim,
                   cutoff=ltensor._cutoff,
                   dtype=ltensor._dtype,
                   idx_dtype=ltensor._idx_dtype,
                   tensors=tensors), norm

    @staticmethod
    def create_diag(*,
                    name: str = None,
                    rank: int = -1,
                    diag: tf.Tensor = None,
                    dtype=None,
                    max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                    cutoff: float = DEFAULT_CUTOFF) -> MPS:
        assert len(diag.shape) == 1
        tensors = list()
        if rank == 1:
            tensors.append(tf.expand_dims(tf.expand_dims(diag, axis=0), axis=-1))
        else:
            tensors.append(tf.expand_dims(custom_diag(2,
                                                      diag,
                                                      dtype=dtype),
                                          axis=0))
            for idx in range(1, rank - 1):
                tensors.append(custom_diag(3,
                                           tf.ones(diag.shape[0]),
                                           dtype=dtype))
            tensors.append(tf.expand_dims(custom_diag(2,
                                                      tf.ones(diag.shape[0]),
                                                      dtype=dtype),
                                          axis=-1))

        return MPS(name=name,
                   visible_num=rank,
                   phys_dims=diag.shape[0],
                   bond_dims=diag.shape[0],
                   given_orth_idx=0,
                   dtype=dtype if dtype is not None else diag.dtype,
                   max_bond_dim=max_bond_dim,
                   cutoff=cutoff if cutoff is not None else cutoff,
                   tensors=tensors)

    @classmethod
    def calc_inversion_num(cls,
                           *,
                           perm: Tuple[int, ...] = None) -> int:
        """
        Calculates the number of inversions in a permutation of n integers ranging from 0 to (n-1).

        :param perm:
        :return:
        """
        result = 0
        for idx in range(len(perm) - 1):
            for jdx in range(idx, len(perm)):
                if perm[idx] > perm[jdx]:
                    result += 1

        return result

    @classmethod
    def calc_positions(cls,
                       *,
                       perm: Tuple[int, ...] = None) -> Tuple[int, ...]:
        """
        Calculates such array positions that perm[positions[idx]] = idx
        :param perm:
        :return:
        """
        return tuple([tup[0] for tup in sorted(enumerate(perm), key=lambda tup: tup[1])])

    @classmethod
    def calc_contraction_perm(cls,
                              *,
                              lindices: Tuple[int, ...] = None,
                              rindices: Tuple[int, ...] = None) -> Tuple[int, ...]:
        """
        lindices and rindices are lists of indices connecting two MPSes, which are being contracted.
        Thus, len(lindices) == len(rindices) =: idx_num.

        This function calculates a permutation required to resolve all crosses after indices
        are sorted in each MPS (in the left one indices are sorted in the ascending order, in the
        right one indices are sorted in the descending order).

        For example, if
        lindices = (0, 2, 3, 8, 5, 6),
        rindices = (7, 6, 5, 4, 2, 0),
        then summed pairs are:
        ((0, 7), (2, 6), (3, 5), (8, 4), (5, 2), (6, 0)).

        We sort pairs according to the rdx and substitute rdx to a range (0, idx_num - 1)
        sort by rdx: ((6, 0), (5, 2), (8, 4), (3, 5), (2, 6), (0, 7))
        substitute: ((6, 0), (5, 1), (8, 2), (3, 3), (2, 4), (0, 5))

        Now, we sort pairs in descending order of ldx and readout the permutation from second element
        of each couple
        sort by -ldx: ((8, 2), (6, 0), (5, 1), (3, 3), (2, 4), (0, 5))
        perm: (2, 0, 1, 3, 4, 5)

        :param lindices:
        :param rindices:
        :return:
        """
        assert len(lindices) == len(rindices)

        summed_pairs = list(zip(lindices, rindices))
        summed_pairs = sorted(summed_pairs, key=lambda tup: tup[1])
        for pos, idx_pair in enumerate(summed_pairs):
            summed_pairs[pos] = idx_pair[0], pos
        summed_pairs = sorted(summed_pairs, key=lambda tup: -tup[0])

        return tuple([tup[1] for tup in summed_pairs])

    @classmethod
    def calc_min_swap_num(cls,
                          *,
                          ltensor: MPS = None,
                          lindices: Tuple[int, ...] = None,
                          rtensor: MPS = None,
                          rindices: Tuple[int, ...] = None) -> int:
        """
        Calculates the minimum required number of swaps to align to MPSes given their relative order
        and all connected indices.

        :param ltensor:
        :param lindices:
        :param rtensor:
        :param rindices:
        :return:
        """
        shift_swap_num = 0
        for pos, ldx in enumerate(sorted(lindices)[::-1]):
            shift_swap_num += (ltensor.rank - 1) - ldx - pos
        for pos, rdx in enumerate(sorted(rindices)):
            shift_swap_num += rdx - pos

        perm = cls.calc_contraction_perm(lindices=lindices, rindices=rindices)

        return shift_swap_num + cls.calc_inversion_num(perm=perm)

    @classmethod
    def resolve_crosses(cls,
                        *,
                        ltensor: MPS = None,
                        lindices: Tuple[int, ...] = None,
                        rtensor: MPS = None,
                        rindices: Tuple[int, ...] = None):
        assert ltensor._dtype == rtensor._dtype
        assert ltensor._idx_dtype == rtensor._idx_dtype
        assert ltensor._max_bond_dim == rtensor._max_bond_dim
        assert ltensor._cutoff == rtensor._cutoff

        assert len(lindices) == len(rindices)
        idx_num = len(lindices)
        # Check that lindices are idx_num rightmost indices of ltensor
        assert np.all(np.asarray(sorted(lindices)) == np.arange(ltensor.rank - idx_num,
                                                                ltensor.rank))
        # Check that rindices are idx_num leftmost indices of rtensor
        assert np.all(np.asarray(sorted(rindices)) == np.arange(0, idx_num))
        # Check that all indices have same dimension
        assert np.all(np.asarray(ltensor.shape)[np.asarray(lindices)]
                      == np.asarray(rtensor.shape)[np.asarray(rindices)])
        perm = cls.calc_contraction_perm(lindices=lindices,
                                         rindices=rindices)

        for perm_idx in range(idx_num):
            positions = cls.calc_positions(perm=perm)
            fdx = positions[perm_idx]
            offset_fdx = ltensor.rank - 1 - fdx
            tdx = perm_idx
            offset_tdx = ltensor.rank - 1 - tdx

            # Permute actual indices
            ltensor.move(fdx=offset_fdx, tdx=offset_tdx)

            # Permute helper indices
            perm = np.asarray(perm)
            if fdx < tdx:
                perm[fdx:tdx] = perm[(fdx + 1):(tdx + 1)]
            else:
                perm[(tdx + 1):(fdx + 1)] = perm[tdx:fdx]
            perm[tdx] = tdx

    @classmethod
    def contract_by_indices(cls,
                            *,
                            lmps: MPS = None,
                            lindices: Tuple[int, ...] = None,
                            rmps: MPS = None,
                            rindices: Tuple[int, ...] = None,
                            name: str = None,
                            new_orth_idx: int = None) -> Tuple[MPS, tf.Tensor, Dict[int, int], Dict[int, int]]:
        assert lmps._dtype == rmps._dtype
        assert lmps._idx_dtype == rmps._idx_dtype
        assert lmps._max_bond_dim == rmps._max_bond_dim
        assert lmps._cutoff == rmps._cutoff

        assert len(lindices) == len(rindices)
        idx_num = len(lindices)
        # Check that all indices have same dimension
        assert np.all(np.asarray(lmps.shape)[np.asarray(lindices)]
                      == np.asarray(rmps.shape)[np.asarray(rindices)])
        lmps._cur_to_old = {ldx: ldx for ldx in range(lmps.rank)}
        rmps._cur_to_old = {rdx: rdx for rdx in range(rmps.rank)}
        summed_pairs = list(zip(lindices, rindices))
        # Group all coupled indices of the left MPS at its right edge
        for pos, (ldx, rdx) in enumerate(sorted(summed_pairs, key=lambda tup: tup[0])[::-1]):
            lmps.move(ldx, lmps.rank - 1 - pos)
            summed_pairs[pos] = (lmps.rank - 1 - pos, rdx)
        for pos, (ldx, rdx) in enumerate(sorted(summed_pairs, key=lambda tup: tup[1])):
            rmps.move(rdx, pos)
            summed_pairs[pos] = (ldx, pos)

        shifted_lindices = tuple([tup[0] for tup in summed_pairs])
        shifted_rindices = tuple([tup[1] for tup in summed_pairs])
        cls.resolve_crosses(ltensor=lmps,
                            lindices=shifted_lindices,
                            rtensor=rmps,
                            rindices=shifted_rindices)

        #TODO: Test if the following canonicalisation is required
        lmps._canonicalise(lmps.rank - 1)
        rmps._canonicalise(0)

        # TODO: Calculate bond tensor in a very straightforward way
        bond_tensor = tf.eye(num_rows=lmps._tensors[lmps.rank - 1].shape[2],
                             num_columns=rmps._tensors[0].shape[0],
                             dtype=lmps._dtype)
        for idx in range(idx_num):
            ldx = lmps.rank - 1 - idx
            rdx = idx

            assert lmps._tensors[ldx].shape[2] == bond_tensor.shape[0]
            bond_tensor = tf.tensordot(lmps._tensors[ldx],
                                       bond_tensor,
                                       axes=[[2], [0]])
            assert bond_tensor.shape[1] == rmps._tensors[rdx].shape[1]
            assert bond_tensor.shape[2] == rmps._tensors[rdx].shape[0]
            bond_tensor = tf.tensordot(bond_tensor,
                                       rmps._tensors[rdx],
                                       axes=[[1, 2], [1, 0]])
        # TODO: Create a new MPS
        tensors = lmps._tensors[:-idx_num] + rmps._tensors[idx_num:]
        if lmps.rank > idx_num:
            tensors[lmps.rank - 1 - idx_num] = tf.einsum('iaj,jk->iak',
                                                         tensors[lmps.rank - 1 - idx_num],
                                                         bond_tensor)
            bond_dims = lmps._bond_dims[:-idx_num] + rmps._bond_dims[idx_num - 1:]
        elif rmps.rank > idx_num:
            tensors[0] = tf.einsum('ij,jak->iak',
                                   bond_tensor,
                                   tensors[0])
            bond_dims = rmps._bond_dims[idx_num:]
        else:
            assert bond_tensor.shape == (1, 1)
            tensors.append(bond_tensor)
            bond_dims = []

        norm = None
        if lmps.rank > idx_num:
            norm = tf.norm(tensors[lmps.rank - 1 - idx_num])
            tensors[lmps.rank - 1 - idx_num] = tf.divide(tensors[lmps.rank - 1 - idx_num], norm)
        else:
            norm = tf.norm(tensors[0])
            tensors[0] = tf.divide(tensors[0], norm)

        name = f'({lmps._name}_{lindices}-{rmps._name}_{rindices})' if name is None else name

        visible_num = (lmps.rank - idx_num) + (rmps.rank - idx_num)
        phys_dims = lmps._phys_dims[:-idx_num] + rmps._phys_dims[idx_num:]

        return MPS(name=name,
                   visible_num=visible_num,
                   phys_dims=phys_dims,
                   bond_dims=bond_dims,
                   given_orth_idx=lmps.rank - 1 - idx_num if lmps.rank > idx_num else 0,
                   new_orth_idx=new_orth_idx if new_orth_idx is not None else None,
                   max_bond_dim=lmps._max_bond_dim,
                   cutoff=lmps._cutoff,
                   dtype=lmps._dtype,
                   idx_dtype=lmps._idx_dtype,
                   tensors=tensors), norm, lmps.old_to_cur(), rmps.old_to_cur()
