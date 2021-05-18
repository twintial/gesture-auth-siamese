from tensorflow.keras import Model, initializers, activations
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D
import tensorflow.keras.layers as l
import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True) # 可以调试layer和model里的call，不用graphic


class CBN(l.Layer):
    def __init__(self, filters, kernel_size, conv_strides, pool_size, pool_strides, conv_padding='valid',
                 pool_padding='same', se=False, **kwargs):
        super().__init__(**kwargs)
        self.l_Conv2D = Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=conv_strides,
                               bias_initializer=initializers.Constant(value=0.1),
                               padding=conv_padding)
        self.l_bn = BatchNormalization()
        self.l_relu = ReLU()
        self.l_mp = MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding)

        self.r = 16  # 之前是4
        self.gap = l.GlobalAveragePooling2D()
        self.d1 = l.Dense(filters // self.r, use_bias=False, activation='relu')
        self.d2 = l.Dense(filters, use_bias=False, activation='sigmoid')

        self.se = se

    def call(self, inputs, **kwargs):
        x = self.l_Conv2D(inputs)
        x = self.l_bn(x)
        x = self.l_relu(x)
        x = self.l_mp(x)
        # senet
        if self.se:
            s = self.gap(x)
            s = self.d1(s)
            s = self.d2(s)
            x = l.Multiply()([x, s])
        return x

    def get_config(self):
        config = super().get_config().copy()
        # config.update({
        #
        # })
        return config


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
    x_p = l.Dense(256, activation='relu', bias_initializer=initializers.Constant(value=0.1))(x_p)
    # magn
    x_m = CBN(8, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(magn_input)
    x_m = CBN(16, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 4), pool_strides=(1, 4))(x_m)
    x_m = CBN(32, kernel_size=(3, 5), conv_strides=(1, 1), pool_size=(2, 3), pool_strides=(2, 3))(x_m)
    x_m = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(x_m)
    x_m = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), pool_strides=(2, 2))(x_m)
    x_m = l.Flatten()(x_m)
    x_m = l.Dense(256, activation='relu', bias_initializer=initializers.Constant(value=0.1))(x_m)
    # concat
    fusion_embedding = l.concatenate([x_p, x_m])
    flattened_fe = l.Flatten()(fusion_embedding)

    flattened_fe = l.Dense(128, activation='relu', bias_initializer=initializers.Constant(value=0.1))(flattened_fe)

    dropout_fe = l.Dropout(0.5)(flattened_fe)
    output = l.Dense(n_classes, activation=activations.get('softmax'))(dropout_fe)
    return Model(inputs=[phase_input, magn_input], outputs=[output])
