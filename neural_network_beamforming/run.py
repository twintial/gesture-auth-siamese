from config import *
from neural_network_beamforming.model import cons_phase_model, cons_magn_model
from neural_network_beamforming.data_loader import DataLoader
from neural_network_beamforming.model import DeepUltraGesture


def main():
    phase_model = cons_phase_model((NUM_OF_FREQ * 2, PADDING_LEN-2, 1))
    magn_model = cons_magn_model((NUM_OF_FREQ * 2, PADDING_LEN-2, 1))
    dug = DeepUltraGesture(phase_model, magn_model, 10)
    # 构建数据集
    dataset_loader = DataLoader([r'D:\实验数据\2021\neuralbeamform\dataset1\random_split\train.tfrecord'],
                                [r'D:\实验数据\2021\neuralbeamform\dataset1\random_split\test.tfrecord'],
                                BATCH_SIZE)
    train_set = dataset_loader.get_train_set()
    test_set = dataset_loader.get_test_set()
    dug.train_with_tfdataset(train_set, test_set)


if __name__ == '__main__':
    main()