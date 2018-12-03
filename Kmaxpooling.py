import numpy as np
import theano.tensor as T
from keras.layers.core import MaskedLayer

class KMaxPooling(MaskedLayer):
    def __init__(self, pooling_size):
        super(MaskedLayer, self).__init__()
        self.pooling_size = pooling_size
        self.input = T.tensor3()

    def get_output_mask(self, train=False):
        return None

    def get_output(self, train=False):
        data = self.get_input(train)
        mask = self.get_input_mask(train)

        if mask is None:
            mask = T.sum(T.ones_like(data), axis=-1)
        mask = mask.dimshuffle(0, 1, "x")

        masked_data = T.switch(T.eq(mask, 0), -np.inf, data)

        result = masked_data[T.arange(masked_data.shape[0]).dimshuffle(0, "x", "x"),
                             T.sort(T.argsort(masked_data, axis=1)[:, -self.pooling_size:, :], axis=1),
                             T.arange(masked_data.shape[2]).dimshuffle("x", "x", 0)]

        return result

    def get_config(self):
        return {"name" : self.__class__.__name__, "pooling_size" : self.pooling_size}
