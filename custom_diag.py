import tensorflow as tf


def custom_diag(rank: int = -1,
                diag: tf.Tensor = None,
                dtype=None) -> tf.Tensor:
    assert len(diag.shape) == 1
    dim = diag.shape[0]
    diag_indices = tf.transpose(tf.tile([tf.range(dim)], (rank, 1)))
    if dtype is not None:
        diag = tf.cast(diag, dtype=dtype)

    return tf.scatter_nd(diag_indices,
                         diag,
                         shape=[dim] * rank)
