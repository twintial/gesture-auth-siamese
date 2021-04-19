from config import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL
from tensorflow.python.keras.utils.vis_utils import plot_model
from siamese_cons_loss.cnn import *
from siamese_cons_loss.phase_pair_loader import PhasePairLoader
from siamese_cons_loss.siamese_net import SiameseNet


def main():
    # 构建siamese
    input_shape = phase_input_shape
    cnn_net = cons_cnn_model(input_shape)
    # cnn_net.summary()
    plot_model(cnn_net, to_file="model/fusion_cnn.png", show_shapes=True)
    # siamese_net = SiameseNet(cnn_net)
    # siamese_net.models.summary()
    # plot_model(siamese_net.model, to_file="siamese_cons_loss.png", show_shapes=True)
    # 构建数据集
    # dataset_loader = PhasePairLoader([r'D:\实验数据\2021\siamese\e1\train_tfrecord\train.tfrecord'],
    #                                  [r'D:\实验数据\2021\siamese\e1\test_tfrecord\test.tfrecord'],
    #                                  batch_size=BATCH_SIZE)
    # train_set = dataset_loader.get_train_set()
    # test_set = dataset_loader.get_test_set()
    # siamese_net.train_with_datasets(train_set, test_set, epochs=1000)
    # siamese_net.save_weights('temple.h5')


def test():
    input_shape = phase_input_shape
    resnet = resnet_34(input_shape)
    plot_model(resnet, to_file="net_png/resnet34.png", show_shapes=True)


if __name__ == '__main__':
    # 没有归一化
    main()
    # test()
