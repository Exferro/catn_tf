import tensorflow as tf

from .constants import BASE_FLOAT_TYPE


class TensorInitialiser:
    FLOAT_TYPES = (tf.float32, tf.float64)
    COMPLEX_TYPES = (tf.complex64, tf.complex128)
    COMPLEX_TO_FLOAT = {
        tf.complex64: tf.float32,
        tf.complex128: tf.float64,
    }

    def __init__(self,
                 *,
                 dtype=BASE_FLOAT_TYPE,
                 init_method=None):
        assert (dtype in self.FLOAT_TYPES) or (dtype in self.COMPLEX_TYPES)
        self._dtype = dtype
        if init_method is None:
            self._init_method = lambda shape, dtype:  0.01 * tf.random.normal(shape=shape, dtype=dtype)
        else:
            self._init_method = init_method

    def __call__(self, shape):
        if self._dtype in TensorInitialiser.COMPLEX_TYPES:
            return tf.complex(self._init_method(shape=shape,
                                                dtype=self.COMPLEX_TO_FLOAT[self._dtype]),
                              self._init_method(shape=shape,
                                                dtype=self.COMPLEX_TO_FLOAT[self._dtype]))
        elif self._dtype in TensorInitialiser.FLOAT_TYPES:
            return self._init_method(shape=shape, dtype=self._dtype)
        else:
            raise RuntimeError(f'At some strange reason TensorInitialiser has reached '
                               f'unreachable piece of code in the __call__ member function')
