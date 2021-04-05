import os
from config import TF_CPP_MIN_LOG_LEVEL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL
import numpy as np
import tensorflow as tf
import log


def tensor_feature(tensor):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tensor).numpy()]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class PhasePairConvertor:
    def __init__(self):
        self.path_label_tuples = None

    def _create_path_label_tuples(self, npz_path):
        dir_list = []
        user_dirs = os.listdir(npz_path)
        for user_dir in user_dirs:
            gesture_dirs = os.listdir(os.path.join(npz_path, user_dir))
            for gesture_dir in gesture_dirs:
                abs_gesture_dir = os.path.join(npz_path, user_dir, gesture_dir)
                dir_list.append(abs_gesture_dir)
        labels = np.arange(len(dir_list))
        self.path_label_tuples = list(zip(dir_list, labels))

    def create_pairs(self, npz_path, target_path, tfrecord_file_name):
        self._create_path_label_tuples(npz_path)
        log.logger.info('=' * 10 + ' start to generate tfrecord ' + '=' * 10)
        positive_pair_num = 0
        negative_pair_num = 0
        os.makedirs(target_path)
        with tf.io.TFRecordWriter(os.path.join(target_path, tfrecord_file_name)) as f:
            tuple_num = len(self.path_label_tuples)
            for tuple_index in range(tuple_num):
                path_label_tuple = self.path_label_tuples[tuple_index]
                # 列出文件夹中所有npz文件名
                npz_file_names = os.listdir(path_label_tuple[0])
                file_num = len(npz_file_names)
                # 遍历每一个npz文件
                for i in range(file_num):
                    data_i = np.load(os.path.join(path_label_tuple[0], npz_file_names[i]))
                    phase_diff_i = data_i['phase_diff']
                    # create positive pairs
                    for j in range(i + 1, file_num):
                        data_j = np.load(os.path.join(path_label_tuple[0], npz_file_names[j]))
                        phase_diff_j = data_j['phase_diff']
                        # get pair
                        positive_pair = [phase_diff_i, phase_diff_j]
                        label = 1
                        # create tf example
                        tf_example = tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    'pair': tensor_feature(tf.constant(positive_pair, dtype=tf.float32)),
                                    'label': int64_feature(label)
                                }
                            ))
                        f.write(tf_example.SerializeToString())
                        positive_pair_num += 1
                    # create nagetive pairs
                    for fake_tuple_index in range(tuple_index + 1, tuple_num):
                        fake_path_label_tuple = self.path_label_tuples[fake_tuple_index]
                        # 列出另一个文件夹（fake）中所有npz文件名
                        fake_npz_file_names = os.listdir(fake_path_label_tuple[0])
                        for fake_npz_file_name in fake_npz_file_names:
                            fake_data = np.load(os.path.join(fake_path_label_tuple[0], fake_npz_file_name))
                            fake_phase_diff = fake_data['phase_diff']
                            negative_pair = [phase_diff_i, fake_phase_diff]
                            label = 0
                            # create tf example
                            tf_example = tf.train.Example(
                                features=tf.train.Features(
                                    feature={
                                        'pair': tensor_feature(tf.constant(negative_pair, dtype=tf.float32)),
                                        'label': int64_feature(label)
                                    }
                                ))
                            f.write(tf_example.SerializeToString())
                            negative_pair_num += 1
                log.logger.debug(f'finish all npzs in path {path_label_tuple[0]}')
        # 写入样本信息
        with open(os.path.join(target_path, f'{tfrecord_file_name.split(".")[0]}-description.txt'), 'w') as f:
            f.write(f'positive pairs: {positive_pair_num}\n')
            f.write(f'negative pairs: {negative_pair_num}\n')
            f.write(f'total pairs: {positive_pair_num + negative_pair_num}\n')
        log_msg = f'finish all paths in path {npz_path} ' \
            f'with {positive_pair_num} positive pairs and {negative_pair_num} negative pairs, ' \
            f'totally {positive_pair_num + negative_pair_num} paris'
        log.logger.info(log_msg)


if __name__ == '__main__':
    convertor = PhasePairConvertor()
    # 要手动新建train_tfrecord和test_tfrecord
    convertor.create_pairs(r'D:\实验数据\2021\siamese\train_npz', r'D:\实验数据\2021\siamese\train_tfrecord', 'train.tfrecord')
    convertor.create_pairs(r'D:\实验数据\2021\siamese\test_npz', r'D:\实验数据\2021\siamese\test_tfrecord', 'test.tfrecord')
