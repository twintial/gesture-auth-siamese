import os

from fusion_model_improve.data_loader import DataLoader
from fusion_model_improve.fusion_model import FusionImproveModel
from fusion_model_improve.models import basic_model

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from config import *


def main():
    # train_file = [r'D:\实验数据\2021\newgesture\tfrecord\sjj\gesture1',
    #               r'D:\实验数据\2021\newgesture\tfrecord\sjj\gesture2']
    # for i in range(5):
    #     train_file.append(rf'D:\实验数据\2021\newgesture\tfrecord\sjj\gesture1_au{i + 1}')
    #     train_file.append(rf'D:\实验数据\2021\newgesture\tfrecord\sjj\gesture2_au{i + 1}')
    # test_file = [r'D:\实验数据\2021\newgesture\tfrecord\sjj\gesture4',
    #              r'D:\实验数据\2021\newgesture\tfrecord\sjj\gesture3']
    # dataset_loader = DataLoader(train_file, test_file, BATCH_SIZE)
    n_classes = 10
    model = basic_model(n_classes, phase_input_shape, phase_input_shape)
    fusion_model = FusionImproveModel(model, n_classes)
    # 构建数据集
    dataset_loader = DataLoader([r'/media/home/shenjj/dataset/newgesture/random_split/10person/train.tfrecord'],
                                [r'/media/home/shenjj/dataset/newgesture/random_split/10person/test.tfrecord'],
                                BATCH_SIZE)
    train_set = dataset_loader.get_train_set()
    test_set = dataset_loader.get_test_set()
    fusion_model.train_with_tfdataset(train_set, test_set, epochs=500, weights_path='models/fusion.h5')


if __name__ == '__main__':
    main()
