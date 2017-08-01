import numpy as np


class DataSet(object):
  def __init__(self, data):
    self._data = self._auto_expand(data)
    self._num_examples = self._data.shape[0]
    self._index_in_epoch = 0
    # Shuffle the data
    perm = np.arange(self._num_examples)
    np.random.shuffle(perm)
    self._data = self._data[perm]

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def data(self):
    return self._data

  def get_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._data = self._data[perm]
      # Start the next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._data[start:end]

  def _auto_expand(self, data):
    r = len(data.shape)
    if r == 2:
      expanded_data = np.expand_dims(data, axis=0)
      return expanded_data
    elif r < 2 or r > 3:
      print('Inappropriate data rank.')
      exit(1)
    else:
      return data
