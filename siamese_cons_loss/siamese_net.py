from config import phase_input_shape
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras import backend as K

from train_log_formatter import print_log


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


@DeprecationWarning
class SiameseNet:
    def __init__(self, base_net: tf.keras.Model, trained_weight_path=None):
        self.base_net = base_net
        self.trained_weight_path = trained_weight_path
        self.training_history = None

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

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

    def train_with_datasets(self, train_set, test_set, epochs=1000):
        # 速度略慢
        self.train_losses.clear()
        self.train_accuracies.clear()
        self.val_losses.clear()
        self.val_accuracies.clear()

        # Stop criteria variables
        best_validation_accuracy = 0.0
        best_accuracy_iteration = 0
        for i in range(epochs):
            print(f'Epoch {i + 1}/{epochs}')

            train_losses_on_epoch = []
            train_acc_on_epoch = []
            val_losses_on_epoch = []
            val_acc_on_epoch = []
            # 每个epoch训练所有数据
            start_time = time.time()
            for X, Y in train_set:
                loss, acc = self.model.train_on_batch([X[:, 0], X[:, 1]], Y)
                train_losses_on_epoch.append(loss)
                train_acc_on_epoch.append(acc)
            train_loss = np.mean(train_losses_on_epoch)
            train_acc = np.mean(train_acc_on_epoch)
            for te_X, te_Y in test_set:
                loss, acc = self.model.evaluate([te_X[:, 0], te_X[:, 1]], te_Y, verbose=0)
                val_losses_on_epoch.append(loss)
                val_acc_on_epoch.append(acc)
            val_loss = np.mean(val_losses_on_epoch)
            val_acc = np.mean(val_acc_on_epoch)
            end_time = time.time()
            print_log(end_time - start_time, train_loss, train_acc, val_loss, val_acc)

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

    def save_weights(self, weights_path):
        self.model.save_weights(weights_path)
