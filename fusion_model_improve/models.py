from tensorflow.keras import Model, initializers, activations
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D
import tensorflow.keras.layers as l
import tensorflow as tf


class CBN(l.Layer):
    def __init__(self, filters, kernel_size, conv_strides, pool_size, pool_strides, conv_padding='valid',
                 pool_padding='same', **kwargs):
        super().__init__(**kwargs)
        self.l_Conv2D = Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=conv_strides,
                               bias_initializer=initializers.Constant(value=0.1),
                               padding=conv_padding)
        self.l_bn = BatchNormalization()
        self.l_relu = ReLU()
        self.l_mp = MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding)

    def call(self, inputs, **kwargs):
        x = self.l_Conv2D(inputs)
        x = self.l_bn(x)
        x = self.l_relu(x)
        x = self.l_mp(x)
        return x


# class BasicModel(Model):
#
#     def __init__(self, n_classes, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.n_classes = n_classes
#
#     def call(self, inputs, training=None, mask=None):
#         phase_input, magn_input = inputs
#         # phase
#         x_p = CBN(8, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(phase_input)
#         x_p = CBN(16, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 4), pool_strides=(1, 4))(x_p)
#         x_p = CBN(32, kernel_size=(3, 5), conv_strides=(1, 1), pool_size=(2, 3), pool_strides=(2, 3))(x_p)
#         x_p = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(x_p)
#         x_p = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), pool_strides=(2, 2))(x_p)
#         x_p = l.Flatten()(x_p)
#         x_p = l.Dense(128, activation='relu', bias_initializer=initializers.Constant(value=0.1))(x_p)
#         # magn
#         x_m = CBN(8, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(magn_input)
#         x_m = CBN(16, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 4), pool_strides=(1, 4))(x_m)
#         x_m = CBN(32, kernel_size=(3, 5), conv_strides=(1, 1), pool_size=(2, 3), pool_strides=(2, 3))(x_m)
#         x_m = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(x_m)
#         x_m = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), pool_strides=(2, 2))(x_m)
#         x_m = l.Flatten()(x_m)
#         x_m = l.Dense(128, activation='relu', bias_initializer=initializers.Constant(value=0.1))(x_m)
#         # concat
#         fusion_embedding = l.concatenate([x_p, x_m])
#         flattened_fe = l.Flatten()(fusion_embedding)
#         dropout_fe = l.Dropout(0.5)(flattened_fe)
#         output = l.Dense(self.n_classes, activation=activations.get('softmax'))(dropout_fe)
#         return output
#
#     def get_config(self):
#         config = {}
#         return config


def basic_model(n_classes, phase_input_shape, magn_input_shape):
    phase_input = l.Input(shape=phase_input_shape)
    magn_input = l.Input(shape=magn_input_shape)
    # phase
    x_p = CBN(8, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(phase_input)
    x_p = CBN(16, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 4), pool_strides=(1, 4))(x_p)
    x_p = CBN(32, kernel_size=(3, 5), conv_strides=(1, 1), pool_size=(2, 3), pool_strides=(2, 3))(x_p)
    x_p = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(x_p)
    x_p = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), pool_strides=(2, 2))(x_p)
    x_p = l.Flatten()(x_p)
    x_p = l.Dense(128, activation='relu', bias_initializer=initializers.Constant(value=0.1))(x_p)
    # magn
    x_m = CBN(8, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(magn_input)
    x_m = CBN(16, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 4), pool_strides=(1, 4))(x_m)
    x_m = CBN(32, kernel_size=(3, 5), conv_strides=(1, 1), pool_size=(2, 3), pool_strides=(2, 3))(x_m)
    x_m = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(x_m)
    x_m = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), pool_strides=(2, 2))(x_m)
    x_m = l.Flatten()(x_m)
    x_m = l.Dense(128, activation='relu', bias_initializer=initializers.Constant(value=0.1))(x_m)
    # concat
    fusion_embedding = l.concatenate([x_p, x_m])
    flattened_fe = l.Flatten()(fusion_embedding)
    dropout_fe = l.Dropout(0.5)(flattened_fe)
    output = l.Dense(n_classes, activation=activations.get('softmax'))(dropout_fe)
    return Model(inputs=[phase_input, magn_input], outputs=[output])
