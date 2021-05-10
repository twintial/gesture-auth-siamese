import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


class CNNLSTM(Model):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        # fusion layers
        self.input_layer = layers.Input(shape=(None, 8, 100, 1))
        self.conv_1 = layers.TimeDistributed(layers.Conv2D(filters=16, kernel_size=(2, 11), strides=(1, 2)))


class SiameseNet(Model):
    def __init__(self, base_net):
        super(SiameseNet, self).__init__()
        self.distant = layers.Lambda(euclidean_distance)
        self.base_net = base_net

    def call(self, inputs, training=None, mask=None):
        input_a, input_b = inputs
        processed_a = self.base_net(input_a)
        processed_b = self.base_net(input_b)
        distant_output = self.distant([processed_a, processed_b])
        return distant_output


if __name__ == '__main__':
    import numpy as np

    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a[..., np.newaxis])
