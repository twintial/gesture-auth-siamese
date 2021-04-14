from config import *
from fusion.data_loader import DataLoader
from fusion.fusion_model import cons_phase_model, FusionModel, cons_magn_model


def main():
    phase_model = cons_phase_model(phase_input_shape)
    magn_model = cons_magn_model(phase_input_shape)
    fusion_model = FusionModel(phase_model, magn_model, 10)
    # 构建数据集
    dataset_loader = DataLoader([r'D:\实验数据\2021\毕设\micarrayspeaker\span_times_split\train.tfrecord'],
                                [r'D:\实验数据\2021\毕设\micarrayspeaker\span_times_split\test.tfrecord'],
                                BATCH_SIZE)
    train_set = dataset_loader.get_train_set()
    test_set = dataset_loader.get_test_set()
    fusion_model.train_with_tfdataset(train_set, test_set, epochs=1000)


if __name__ == '__main__':
    main()