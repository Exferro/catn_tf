import tensorflow as tf

from tensorflow import int32 as BASE_INT_TYPE
from tensorflow import int32 as BASE_UINT_TYPE
from tensorflow import float64 as BASE_FLOAT_TYPE
from tensorflow import complex128 as BASE_COMPLEX_TYPE

INT_32_TYPE = tf.int32
INT_64_TYPE = tf.int64

UINT_32_TYPE = tf.uint32
UINT_64_TYPE = tf.int64

SUPPORTED_BIT_DEPTHS = (32, 64)

INT_TYPE_TO_BIT_DEPTH = {
    INT_32_TYPE: 32,
    INT_64_TYPE: 64,
}
BIT_DEPTH_TO_INT_TYPE = {
    32: INT_32_TYPE,
    64: INT_64_TYPE,
}
BIT_DEPTH_TO_UINT_TYPE = {
    32: INT_32_TYPE,
    64: INT_64_TYPE,
}

TYPE_TO_BYTES = {
    tf.int8: 1,
    tf.int16: 2,
    tf.int32: 4,
    tf.int64: 8,
    tf.uint32: 4,
    tf.uint64: 8,
    tf.float32: 4,
    tf.float64: 8,
    tf.complex64: 8,
    tf.complex128: 16,
    'complex64': 8,
    'complex128': 16,
}

ALPHA_MASK = tf.constant(0x0aaaaaaaaaaaaaaa, dtype=BASE_UINT_TYPE)
BETA_MASK = tf.constant(0x0555555555555555, dtype=BASE_UINT_TYPE)

CHEMICAL_ACCURACY = 1.6e-3
