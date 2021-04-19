import time

from tensorflow.keras import initializers, layers, activations, Model, optimizers, losses, metrics
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np

from config import *
from train_log_formatter import print_status_bar_ver0


class DeepUltraGesture:
    def __init__(self, phase_model: Model, magn_model: Model, n_classes, trained_weight_path=None):
        self.trained_weight_path = trained_weight_path

        self.phase_model = phase_model
        self.magn_model = magn_model
        # I & Q
        self.I_Q_input_shape = [(PADDING_LEN, N_CHANNELS * NUM_OF_FREQ), (PADDING_LEN, N_CHANNELS * NUM_OF_FREQ)]
        # phase & magn
        self.p_m_input_shape = [phase_input_shape, phase_input_shape]

        # layers for neural network beamforming
        self.blstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))  # 输出长度为512
        self.I_linears = []
        self.Q_linears = []
        for i in range(N_CHANNELS):
            self.I_linears.append(layers.TimeDistributed(layers.Dense(NUM_OF_FREQ, activation=activations.tanh)))
            self.Q_linears.append(layers.TimeDistributed(layers.Dense(NUM_OF_FREQ, activation=activations.tanh)))

        # layers for fusion embedding
        self.f_embedding_fatten = layers.Flatten()
        self.f_embedding_dropout = layers.Dropout(0.5)
        self.f_embedding_softmax = layers.Dense(n_classes, activation=activations.get('softmax'))

        self.beamforming_model = self._construct_beamforming_architecture()
        self.fusion_model = self._construct_fusion_architecture()

    def _construct_beamforming_architecture(self):
        I_input_shape, Q_input_shape = self.I_Q_input_shape
        I_input = layers.Input(shape=I_input_shape)  # batch, PADDING_LEN, N_CHANNELS * NUM_OF_FREQ
        Q_input = layers.Input(shape=Q_input_shape)
        concat_input = layers.concatenate([I_input, Q_input])  # batch, PADDING_LEN, N_CHANNELS * NUM_OF_FREQ * 2
        blstm_output = self.blstm(concat_input)  # batch, PADDING_LEN, 512
        I_outputs = np.zeros(8)
        Q_outputs = np.zeros(8)
        for i in range(N_CHANNELS):
            I_outputs[i] = self.I_linears[i](blstm_output)  # batch, PADDING_LEN, NUM_OF_FREQ
            Q_outputs[i] = self.Q_linears[i](blstm_output)
        I_mask = layers.concatenate(I_outputs)
        Q_mask = layers.concatenate(Q_outputs)
        model = Model(inputs=[I_input, Q_input], outputs=[I_mask, Q_mask])
        return model

    def _construct_fusion_architecture(self):
        phase_input_shape, magn_input_shape = self.p_m_input_shape

        phase_input = layers.Input(shape=phase_input_shape)
        phase_embedding = self.phase_model(phase_input)
        magn_input = layers.Input(shape=magn_input_shape)
        magn_embedding = self.magn_model(magn_input)
        # fusion，最简单的concat
        fusion_embedding = layers.concatenate([phase_embedding, magn_embedding])
        flattened_fe = self.f_embedding_fatten(fusion_embedding)
        dropout_fe = self.f_embedding_dropout(flattened_fe)
        output = self.f_embedding_softmax(dropout_fe)

        model = Model(inputs=[phase_input, magn_input], outputs=[output])
        return model

    def train_with_tfdataset(self, train_set, test_set=None, epochs=1000, weights_path=None):
        mean_train_loss = metrics.Mean(name='loss')
        mean_train_acc = metrics.Mean(name='acc')
        mean_test_loss = metrics.Mean(name='val_loss')
        mean_test_acc = metrics.Mean(name='val_acc')
        best_test_acc = 0  # 记录最好测试集数据
        for epoch in range(epochs):
            # reset states
            mean_train_loss.reset_states()
            mean_train_acc.reset_states()
            mean_test_loss.reset_states()
            mean_test_acc.reset_states()
            print("Epoch {}/{}".format(epoch + 1, epochs))
            start_time = time.time()
            for X, Y in train_set:
                I, Q = X[:, 0], X[:, 1]  # batch, PADDING_LEN, N_CHANNELS * NUM_OF_FREQ
                I_mask, Q_mask = self.beamforming_model([I, Q])  # batch, PADDING_LEN, N_CHANNELS * NUM_OF_FREQ
                I_beamform = I * I_mask - Q * Q_mask
                Q_beamform = I * Q_mask + Q * I_mask
                I_beamform = tf.reshape(I_beamform, (-1, PADDING_LEN, NUM_OF_FREQ, N_CHANNELS))
                Q_beamform = tf.reshape(Q_beamform, (-1, PADDING_LEN, NUM_OF_FREQ, N_CHANNELS))
                I_beamform = tf.reduce_sum(I_beamform, axis=-1)  # batch, PADDING_LEN, NUM_OF_FREQ
                Q_beamform = tf.reduce_sum(Q_beamform, axis=-1)  # batch, PADDING_LEN, NUM_OF_FREQ

            end_time = time.time()
            print_status_bar_ver0(end_time - start_time,
                                  mean_train_loss, mean_train_acc, mean_test_loss, mean_test_acc,
                                  best_acc=best_test_acc)

    # def save_weights(self, weights_path):
    #     self.model.save_weights(weights_path)