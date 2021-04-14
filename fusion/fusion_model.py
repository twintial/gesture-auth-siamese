import time

from tensorflow.keras import initializers, layers, activations, Model, optimizers,losses, metrics
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.models import Sequential

from config import phase_input_shape
from train_log_formatter import print_status_bar_ver0


def cons_phase_model(input_shape):
    cnn_model = Sequential(name='phase_5_layer_CNN')
    cnn_model.add(Conv2D(8,
                         kernel_size=(3, 8),
                         strides=(1, 1),
                         input_shape=input_shape,
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_1'))
    cnn_model.add(BatchNormalization(name='BN_1'))
    cnn_model.add(ReLU(name='Relu_1'))
    cnn_model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same', name='MP_1'))

    cnn_model.add(Conv2D(16,
                         kernel_size=(3, 8),
                         strides=(1, 1),
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_2'))
    cnn_model.add(BatchNormalization(name='BN_2'))
    cnn_model.add(ReLU(name='Relu_2'))
    cnn_model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same', name='MP_2'))

    cnn_model.add(Conv2D(32,
                         kernel_size=(3, 5),
                         strides=(1, 1),
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_3'))
    cnn_model.add(BatchNormalization(name='BN_3'))
    cnn_model.add(ReLU(name='Relu_3'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 3), strides=(2, 3), padding='same', name='MP_3'))

    cnn_model.add(Conv2D(32,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_4'))
    cnn_model.add(BatchNormalization(name='BN_4'))
    cnn_model.add(ReLU(name='Relu_4'))
    cnn_model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same', name='MP_4'))

    cnn_model.add(Conv2D(32,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_5'))
    cnn_model.add(BatchNormalization(name='BN_5'))
    cnn_model.add(ReLU(name='Relu_5'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='MP_5'))

    cnn_model.add(Flatten(name='Flatten'))
    cnn_model.add(Dense(128, activation='relu', bias_initializer=initializers.Constant(value=0.1), name='Dense_1'))
    return cnn_model


def cons_magn_model(input_shape):
    cnn_model = Sequential(name='magn_5_layer_CNN')
    cnn_model.add(Conv2D(8,
                         kernel_size=(3, 8),
                         strides=(1, 1),
                         input_shape=input_shape,
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_1'))
    cnn_model.add(BatchNormalization(name='BN_1'))
    cnn_model.add(ReLU(name='Relu_1'))
    cnn_model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same', name='MP_1'))

    cnn_model.add(Conv2D(16,
                         kernel_size=(3, 8),
                         strides=(1, 1),
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_2'))
    cnn_model.add(BatchNormalization(name='BN_2'))
    cnn_model.add(ReLU(name='Relu_2'))
    cnn_model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same', name='MP_2'))

    cnn_model.add(Conv2D(32,
                         kernel_size=(3, 5),
                         strides=(1, 1),
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_3'))
    cnn_model.add(BatchNormalization(name='BN_3'))
    cnn_model.add(ReLU(name='Relu_3'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 3), strides=(2, 3), padding='same', name='MP_3'))

    cnn_model.add(Conv2D(32,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_4'))
    cnn_model.add(BatchNormalization(name='BN_4'))
    cnn_model.add(ReLU(name='Relu_4'))
    cnn_model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same', name='MP_4'))

    cnn_model.add(Conv2D(32,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_5'))
    cnn_model.add(BatchNormalization(name='BN_5'))
    cnn_model.add(ReLU(name='Relu_5'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='MP_5'))

    cnn_model.add(Flatten(name='Flatten'))
    cnn_model.add(Dense(128, activation='relu', bias_initializer=initializers.Constant(value=0.1), name='Dense_1'))
    return cnn_model


class FusionModel:
    def __init__(self, phase_model: Model, magn_model: Model, n_classes, trained_weight_path=None):
        self.trained_weight_path = trained_weight_path

        self.phase_model = phase_model
        self.magn_model = magn_model
        self.input_shape = [phase_input_shape, phase_input_shape]

        # layers for fusion embedding
        self.f_embedding_fatten = layers.Flatten()
        self.f_embedding_softmax = layers.Dense(n_classes, activation=activations.get('softmax'))

        self.model = self._construct_fusion_architecture()
        self._compile_siamese_model()

    def _construct_fusion_architecture(self):
        phase_input_shape, magn_input_shape = self.input_shape

        phase_input = layers.Input(shape=phase_input_shape)
        phase_embedding = self.phase_model(phase_input)
        magn_input = layers.Input(shape=magn_input_shape)
        magn_embedding = self.magn_model(magn_input)
        # fusion，最简单的concat
        fusion_embedding = layers.concatenate([phase_embedding, magn_embedding])
        flattened_fe = self.f_embedding_fatten(fusion_embedding)
        output = self.f_embedding_softmax(flattened_fe)

        model = Model(inputs=[phase_input, magn_input], outputs=[output])
        return model

    def _compile_siamese_model(self):
        if self.trained_weight_path is not None:
            self.model.load_weights(self.trained_weight_path)
        self.model.compile(loss=losses.sparse_categorical_crossentropy,
                           optimizer=optimizers.Adam(),
                           metrics=metrics.sparse_categorical_accuracy)

    def train_with_tfdataset(self, train_set, test_set=None, epochs=1000):
        mean_train_loss = metrics.Mean(name='loss')
        mean_train_acc = metrics.Mean(name='acc')
        mean_test_loss = metrics.Mean(name='val_loss')
        mean_test_acc = metrics.Mean(name='val_acc')
        for epoch in range(epochs):
            # reset states
            mean_train_loss.reset_states()
            mean_train_acc.reset_states()
            print("Epoch {}/{}".format(epoch + 1, epochs))
            start_time = time.time()
            for X, Y in train_set:
                loss, acc = self.model.train_on_batch([X[:, 0], X[:, 1]], Y)
                mean_train_loss(loss)
                mean_train_acc(acc)
            if test_set is not None:
                for te_X, te_Y in test_set:
                    loss, acc = self.model.train_on_batch([te_X[:, 0], te_X[:, 1]], te_Y)
                    mean_test_loss(loss)
                    mean_test_acc(acc)
            end_time = time.time()
            print_status_bar_ver0(end_time-start_time, mean_train_loss, mean_train_acc, mean_test_loss, mean_test_acc)

    def save_weights(self, weights_path):
        self.model.save_weights(weights_path)
