from tensorflow.keras import initializers, layers, activations, models, regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.models import Sequential


class ResidualUnit(layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = activations.get(activation)
        self.main_layers = [
            layers.Conv2D(filters, 3, strides=strides,
                          padding="same", use_bias=False),
            layers.BatchNormalization(),
            self.activation,
            layers.Conv2D(filters, 3, strides=1,
                          padding="same", use_bias=False),
            layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                layers.Conv2D(filters, 1, strides=strides,
                              padding="same", use_bias=False),
                layers.BatchNormalization()]

    def call(self, inputs, **kwargs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


def resnet_34(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, 7, strides=2, input_shape=input_shape,
                                  padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
    model.add(layers.GlobalAvgPool2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    return model


def cons_cnn_model(input_shape):
    cnn_model = Sequential(name='5_layer_CNN')
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
    cnn_model.add(Dropout(rate=0.2, name='Dropout'))  # 表示丢弃,0.5还行
    cnn_model.add(Dense(64, activation='softmax', name='Output_layer'))
    return cnn_model


def cons_cnn_model_gai(input_shape):
    cnn_model = Sequential(name='5_layer_CNN')
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
    cnn_model.add(Dense(128, bias_initializer=initializers.Constant(value=0.1), name='Dense_1'))
    return cnn_model
