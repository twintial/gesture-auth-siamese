from config import *
from siamese_cons_loss.cnn import *
from siamese_cons_loss.phase_pair_loader import PhasePairLoader
from tripet_loss.model import TripLossModel
from tripet_loss.phase_data_loader import PhaseDataLoader
import numpy as np

from tripet_loss.siamese_custom import Siam


# tirplet loss
def main():
    np.random.seed(10)
    input_shape = phase_input_shape
    cnn_net = cons_cnn_model_gai(input_shape)
    model = TripLossModel(cnn_net, input_shape, 10, 10, 0.5)
    # data_loader = PhaseDataLoader([r'D:\实验数据\2021\siamese\e1\train_npz'])

    data_loader = PhaseDataLoader([r'D:\实验数据\2021\siamese\newgesture\npz'])

    loader = PhasePairLoader([r'D:\实验数据\2021\siamese\e1\train_tfrecord\train.tfrecord'],
                             [r'D:\实验数据\2021\siamese\e1\test_tfrecord\test.tfrecord'], BATCH_SIZE)
    model.train(data_loader, loader.get_test_set(), steps=10, epochs=50)


# 对比损失
def test_siam():
    np.random.seed(10)
    input_shape = phase_input_shape
    cnn_net = cons_cnn_model_gai(input_shape)
    model = Siam(cnn_net, input_shape, 0.5)  # 0.5可能更好，这个margin类似学习率，降低了过拟合率
    loader = PhasePairLoader([r'D:\实验数据\2021\siamese\e1\train_tfrecord\train.tfrecord'],
                             [r'D:\实验数据\2021\siamese\e1\test_tfrecord\test.tfrecord'], BATCH_SIZE)
    model.train(loader.get_train_set(), loader.get_test_set())


# triplet loss 或 对比损失 evaluate
def eval_siam():
    np.random.seed(10)
    input_shape = phase_input_shape
    cnn_net = cons_cnn_model_gai(input_shape)
    # cnn_net = resnet_34(input_shape)
    model = Siam(cnn_net, input_shape, 0.5)
    loader = PhasePairLoader([r'D:\实验数据\2021\siamese\e1\train_tfrecord\train.tfrecord'],
                             [r'D:\实验数据\2021\siamese\e2\test_tfrecord\test.tfrecord',
                              r'D:\实验数据\2021\siamese\e3\test_tfrecord\test.tfrecord',
                              r'D:\实验数据\2021\siamese\e4\test_tfrecord\test.tfrecord',
                              r'D:\实验数据\2021\siamese\e5\test_tfrecord\test.tfrecord'], BATCH_SIZE)
    model.evaluate(loader.get_test_set(), weights_path='models/siamese.h5')


import tensorflow.keras.backend as K


def triplet_loss(y_pred):
    assert y_pred.shape[0] % 3 == 0
    anchor = y_pred[::3]
    pos = y_pred[1::3]
    neg = y_pred[2::3]
    pos_dist = K.sum(K.square(anchor - pos), axis=1)
    neg_dist = K.sum(K.square(anchor - neg), axis=1)
    basic_loss = pos_dist - neg_dist + 0.3
    loss = K.mean(K.maximum(-basic_loss, 0))

    return loss


if __name__ == '__main__':
    # import tensorflow as tf
    #
    # a = 0.1
    # t1 = a * (-6-2.4)
    # t2 = a * (-6-6)
    # t3 = a * (12+8)
    # w1 = tf.Variable([[5. - t1, 6. - t1, 7. - t1], [8. - t2, 9. - t2, 10. - t2], [11. - t3, 12. - t3, 13. - t3]])
    # with tf.GradientTape() as tape:
    #     z = triplet_loss(w1)
    # print(z)
    # gradients = tape.gradient(z, [w1])
    # print(gradients)
    main()
    # test_siam()
    # eval_siam()
