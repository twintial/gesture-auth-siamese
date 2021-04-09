import tensorflow as tf
# for gpu in tf.config.experimental.list_physical_devices("GPU"):
#     tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import layers, Model, optimizers, metrics
import tensorflow.keras.backend as K
import numpy as np
import time
import log

from train_log_formatter import print_status_bar


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def cal_acc(y_true, y_pred, threshold):
    return np.mean((tf.cast(y_pred, tf.float32) < threshold) == tf.cast(y_true, tf.bool))


def accuracy(y_true, y_pred, threshold):  # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))

# batch的情况下不太好用
def calculate_val_far(y_true, y_pred, threshold):
    # y_pred等同于dist
    predict_is_same = K.cast(y_pred < threshold, y_true.dtype)
    true_accept = np.sum(np.logical_and(predict_is_same, y_true))
    false_accept = np.sum(np.logical_and(predict_is_same, np.logical_not(y_true)))
    n_same = np.sum(y_true)
    n_diff = np.sum(np.logical_not(y_true))
    val = true_accept / n_same
    far = false_accept / n_diff
    return val, far


def calculate_tp_fp(y_true, y_pred, threshold):
    # y_pred等同于dist
    predict_is_same = K.cast(y_pred < threshold, y_true.dtype)
    true_accept = np.sum(np.logical_and(predict_is_same, y_true))
    false_accept = np.sum(np.logical_and(predict_is_same, np.logical_not(y_true)))
    return true_accept, false_accept



class Siam:
    def __init__(self, backbone, input_shape, margin):
        self.backbone: Model = backbone
        self.input_shape = input_shape
        self.margin = margin

        self.optimizer = optimizers.Adam()

        self.l2_norm = layers.Lambda(lambda embeddings: K.l2_normalize(embeddings, axis=1), name='l2_norm')
        self.model = self._construct_model(self.input_shape)

    def _construct_model(self, input_shape):
        input = layers.Input(shape=input_shape)
        # size = class_per_batch * phases_per_class
        embeddings = self.backbone(input)
        embeddings = self.l2_norm(embeddings)
        model = Model(inputs=input, outputs=embeddings)
        return model

    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        y_true = tf.cast(y_true, tf.float32)
        sqaure_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

    def train(self, train_dataset, val_dataset=None, epochs=1000):
        mean_train_loss = metrics.Mean(name='loss')
        mean_train_acc = metrics.Mean(name='acc')
        for epoch in range(epochs):
            # reset states
            mean_train_loss.reset_states()
            mean_train_acc.reset_states()

            print("Epoch {}/{}".format(epoch + 1, epochs))
            start_time = time.time()
            for X, Y in train_dataset:
                with tf.GradientTape() as tape:
                    embeddings1 = self.model(X[:, 0], training=True)
                    embeddings2 = self.model(X[:, 1], training=True)
                    dist = euclidean_distance([embeddings1, embeddings2])
                    constractive_loss = self.contrastive_loss(Y, dist)
                    # self.model.losses是一个数组，可能是一个空数组，和l2正则化之类的有关
                    loss = tf.add_n([constractive_loss] + self.model.losses)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                mean_train_loss(loss)
                mean_train_acc(accuracy(Y, dist, 0.5))
                # evaluate
            if val_dataset is not None:
                thresholds = np.arange(0, 4, 0.01)
                nof_thresholds = len(thresholds)
                acc_test = np.zeros(nof_thresholds)
                tp_test = np.zeros(nof_thresholds)
                fp_test = np.zeros(nof_thresholds)
                n_same, n_diff = 0, 0
                total_batch = 0
                for te_X, te_Y in val_dataset:
                    embeddings1 = self.model(te_X[:, 0], training=False)
                    embeddings2 = self.model(te_X[:, 1], training=False)
                    dist = euclidean_distance([embeddings1, embeddings2])
                    n_same += np.sum(te_Y)
                    n_diff += np.sum(np.logical_not(te_Y))
                    for threshold_idx, threshold in enumerate(thresholds):
                        acc_test[threshold_idx] += accuracy(te_Y, dist, threshold)
                        tp, fp = calculate_tp_fp(te_Y, dist, threshold)
                        tp_test[threshold_idx] += tp
                        fp_test[threshold_idx] += fp
                    total_batch += 1
                acc_test /= total_batch
                # 获得最好threshold
                best_threshold_index = np.argmax(acc_test)
                best_threshold = thresholds[best_threshold_index]
                best_acc = acc_test[best_threshold_index]
                best_val = tp_test[best_threshold_index] / n_same
                best_far = fp_test[best_threshold_index] / n_diff
                print(f'- best_threshold: {best_threshold} - best_acc: {best_acc:.4f} - best_val: {best_val:.4f} - best_far: {best_far:.4f}')

            end_time = time.time()
            log_str = f'- {end_time - start_time:.0f}s - {mean_train_loss.name}:{mean_train_loss.result():.4f} - {mean_train_acc.name}:{mean_train_acc.result():.4f}'
            print(log_str)
