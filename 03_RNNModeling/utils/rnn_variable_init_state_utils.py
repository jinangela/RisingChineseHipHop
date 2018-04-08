import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape
from enum import Enum


def _state_size_with_prefix(state_size, prefix=None):
    """Helper function that enables int or TensorShape shape specification.
    This function takes a size specification, which can be an integer or a
    TensorShape, and converts it into a list of integers. One may specify any
    additional dimensions that precede the final state size specification.
    Args:
        state_size: TensorShape or int that specifies the size of a tensor.
        prefix: optional additional list of dimensions to prepend.
    Returns:
        result_state_size: list of dimensions the resulting tensor size.
    """
    result_state_size = tensor_shape.as_shape(state_size).as_list()
    if prefix is not None:
        if not isinstance(prefix, list):
            raise TypeError("prefix of _state_size_with_prefix should be a list.")
        result_state_size = prefix + result_state_size
    return result_state_size


def get_initial_cell_state(cell, initializer, batch_size, dtype):
    """Return state tensor(s), initialized with initializer.
    Args:
      cell: RNNCell.
      batch_size: int, float, or unit Tensor representing the batch size.
      initializer: function with two arguments, shape and dtype, that
          determines how the state is initialized.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` initialized
      according to the initializer.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = cell.state_size
    if nest.is_sequence(state_size):
        state_size_flat = nest.flatten(state_size)
        init_state_flat = [initializer(_state_size_with_prefix(s), batch_size, dtype, i)
                           for i, s in enumerate(state_size_flat)]
        init_state = nest.pack_sequence_as(structure=state_size, flat_sequence=init_state_flat)
    else:
        init_state_size = _state_size_with_prefix(state_size)
        init_state = initializer(init_state_size, batch_size, dtype, None)
    return init_state


def zero_state_initializer(shape, batch_size, dtype, index):
    z = tf.zeros(tf.stack(_state_size_with_prefix(shape, [batch_size])), dtype)
    z.set_shape(_state_size_with_prefix(shape, prefix=[None]))
    return z


def make_variable_state_initializer(**kwargs):
    def variable_state_initializer(shape, batch_size, dtype, index):
        args = kwargs.copy()

        if args.get('name'):
            args['name'] = args['name'] + '_' + str(index)
        else:
            args['name'] = 'init_state_' + str(index)

        args['shape'] = shape
        args['dtype'] = dtype

        var = tf.get_variable(**args)
        var = tf.expand_dims(var, 0)
        var = tf.tile(var, tf.stack([batch_size] + [1] * len(shape)))
        var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
        return var
    return variable_state_initializer


def make_gaussian_state_initializer(initializer, deterministic_tensor=None, stddev=0.3):
    def gaussian_state_initializer(shape, batch_size, dtype, index):
        init_state = initializer(shape, batch_size, dtype, index)
        if deterministic_tensor is not None:
            return tf.cond(deterministic_tensor,
                           lambda: init_state,
                           lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev))
        else:
            return init_state + tf.random_normal(tf.shape(init_state), stddev=stddev)
    return gaussian_state_initializer


class StateInitializer(Enum):
    ZERO_STATE = 1
    VARIABLE_STATE = 2
    NOISY_ZERO_STATE = 3
    NOISY_VARIABLE_STATE = 4
