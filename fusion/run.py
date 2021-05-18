import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from config import *
from fusion.data_loader import DataLoader
from fusion.fusion_model import cons_phase_model, FusionModel, cons_magn_model



def main():
    # train_file = [r'D:\实验数据\2021\newgesture\tfrecord\sjj\gesture1',
    #               r'D:\实验数据\2021\newgesture\tfrecord\sjj\gesture2']
    # for i in range(5):
    #     train_file.append(rf'D:\实验数据\2021\newgesture\tfrecord\sjj\gesture1_au{i + 1}')
    #     train_file.append(rf'D:\实验数据\2021\newgesture\tfrecord\sjj\gesture2_au{i + 1}')
    # test_file = [r'D:\实验数据\2021\newgesture\tfrecord\sjj\gesture4',
    #              r'D:\实验数据\2021\newgesture\tfrecord\sjj\gesture3']
    # dataset_loader = DataLoader(train_file, test_file, BATCH_SIZE)

    phase_model = cons_phase_model(phase_input_shape)
    magn_model = cons_magn_model(phase_input_shape)
    fusion_model = FusionModel(phase_model, magn_model, 10)
    # # 构建数据集
    # dataset_loader = DataLoader([r'D:\实验数据\2021\newgesture\random_split\10person\train.tfrecord'],
    #                             [r'D:\实验数据\2021\newgesture\random_split\10person\test.tfrecord'],
    #                             BATCH_SIZE)
    # 构建数据集
    dataset_loader = DataLoader([r'/media/home/shenjj/dataset/newgesture/random_split/10person/train.tfrecord'],
                                [r'/media/home/shenjj/dataset/newgesture/random_split/10person/test.tfrecord'],
                                BATCH_SIZE)
    train_set = dataset_loader.get_train_set()
    test_set = dataset_loader.get_test_set()
    fusion_model.train_with_tfdataset(train_set, test_set, epochs=500, weights_path='models/fusion.h5')


if __name__ == '__main__':
    main()
