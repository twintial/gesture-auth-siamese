from config import *
from tensorflow.python.keras.utils.vis_utils import plot_model
from siamese.cnn import cons_cnn_model
from siamese.phase_pair_loader import PhasePairLoader
from siamese.siamese_net import SiameseNet


def main():
    # 构建siamese
    input_shape = phase_input_shape
    cnn_net = cons_cnn_model(input_shape)
    siamese_net = SiameseNet(cnn_net)
    # siamese_net.model.summary()
    # plot_model(siamese_net.model, to_file="siamese.png", show_shapes=True)
    # 构建数据集
    dataset_loader = PhasePairLoader([r'D:\实验数据\2021\siamese\train_tfrecord\train.tfrecord'],
                                     [r'D:\实验数据\2021\siamese\test_tfrecord\test.tfrecord'],
                                     batch_size=BATCH_SIZE)
    train_set = dataset_loader.get_train_set()
    test_set = dataset_loader.get_test_set()
    siamese_net.train_with_datasets(train_set, test_set, epochs=1000)
    siamese_net.save_weights('temple.h5')


if __name__ == '__main__':
    # 没有归一化
    main()
