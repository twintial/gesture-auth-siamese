import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras import backend as K

from config import phase_input_shape


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    y_true = tf.cast(y_true, tf.float32)
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def accuracy(y_true, y_pred):  # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def compute_accuracy(y_true, y_pred):  # numpy上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


class SiameseNet:
    def __init__(self, base_net: tf.keras.Model, trained_weight_path=None):
        self.base_net = base_net
        self.trained_weight_path = trained_weight_path
        self.training_history = None

        self.distant = layers.Lambda(euclidean_distance, name='euclidean_distance')
        self.input_shape = [phase_input_shape, phase_input_shape]
        self.model = self._construct_siamese_architecture(self.input_shape)
        self._compile_siamese_model()

    def _construct_siamese_architecture(self, inputs_shape):
        input_shape_a, input_shape_b = inputs_shape
        input_a = layers.Input(shape=input_shape_a)
        input_b = layers.Input(shape=input_shape_b)

        processed_a = self.base_net(input_a)
        processed_b = self.base_net(input_b)

        distant_output = self.distant([processed_a, processed_b])
        model = Model(inputs=[input_a, input_b], outputs=distant_output)
        return model

    def _compile_siamese_model(self):
        if self.trained_weight_path is not None:
            self.model.load_weights(self.trained_weight_path)
        self.model.compile(loss=contrastive_loss, optimizer=optimizers.Adam(), metrics=[accuracy])

    def train(self, tr_pairs, tr_y, te_pairs, te_y, batch_size=32, epochs=1000, verbose=2):
        self.training_history = self.model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                                               batch_size=batch_size, epochs=epochs, verbose=verbose,
                                               validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

        # compute final accuracy on training and test_demo sets
        y_pred = self.model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
        tr_acc = compute_accuracy(tr_y, y_pred)
        y_pred = self.model.predict([te_pairs[:, 0], te_pairs[:, 1]])
        te_acc = compute_accuracy(te_y, y_pred)

        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        print('* Accuracy on test_demo set: %0.2f%%' % (100 * te_acc))

    def train_with_datasets(self, train_set, test_set, epochs=1000, verbose=2):
        self.training_history = self.model.fit(train_set,
                                               epochs=epochs, verbose=verbose,
                                               validation_data=test_set)

        # compute final accuracy on training and test_demo sets
        self.model.evaluate(train_set)
        self.model.evaluate(test_set)

    def save_weights(self, weights_path):
        self.model.save_weights(weights_path)

