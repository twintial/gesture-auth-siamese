"""
将npz数据按需分割成train set和test set，并保存成tfrecord格式
"""
import os
import re
import log
import numpy as np
import tensorflow as tf


def tensor_feature(tensor):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tensor).numpy()]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class NpzDataSplitor:
    def __init__(self, countinue_same_gesture_num=10):
        self.user_gesture_dict = None
        self.csgn = countinue_same_gesture_num

    @staticmethod
    def npz2tfrecord(target_path, tfrecord_file_name, data_set):
        with tf.io.TFRecordWriter(os.path.join(target_path, tfrecord_file_name)) as f:
            for train_sample in data_set:
                sample_path = train_sample[0]
                sample_label = train_sample[1]
                data = np.load(sample_path)
                sample = [data['phase_diff'], data['magn_diff']]
                tf_example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'phase_magn_diff': tensor_feature(tf.constant(sample, dtype=tf.float32)),
                            'label': int64_feature(sample_label)
                        }
                    ))
                f.write(tf_example.SerializeToString())
        with open(os.path.join(target_path, f'{tfrecord_file_name.split(".")[0]}-description.txt'), 'w') as f:
            f.write(f'sample number : {len(data_set)}\n')

    def _create_filename_label_tuples(self, npz_path):
        self.user_gesture_dict = {}
        user_dirs = os.listdir(npz_path)
        for user_dir in user_dirs:
            filename_label_tuples = []
            gesture_dirs = os.listdir(os.path.join(npz_path, user_dir))
            for gesture_dir in gesture_dirs:
                abs_gesture_dir = os.path.join(npz_path, user_dir, gesture_dir)
                gesture_filenames = os.listdir(abs_gesture_dir)
                for gesture_filename in gesture_filenames:
                    m = re.match(r'(\d+).npz', gesture_filename)
                    if m:
                        code = int(m.group(1))
                        label = code // self.csgn
                        abs_filepath = os.path.join(abs_gesture_dir, gesture_filename)
                        filename_label_tuples.append([abs_filepath, label])
                    else:
                        log.logger.warning(f'unknown filename format {gesture_filename}')
            self.user_gesture_dict[user_dir] = filename_label_tuples

    def random_split_of_per_user(self, npz_path, target_path, train_tfrecord, test_tfrecord, train_set_rate=0.8):
        os.makedirs(target_path)
        train_set = []
        test_set = []
        if self.user_gesture_dict is None:
            self._create_filename_label_tuples(npz_path)
        # 划分
        for v in self.user_gesture_dict.values():
            np.random.shuffle(v)
            v_len = int(len(v))
            v_train_set = v[:int(v_len * train_set_rate)]
            v_test_set = v[int(v_len * train_set_rate):]
            train_set += v_train_set
            test_set += v_test_set
        self.npz2tfrecord(target_path, train_tfrecord, train_set)
        log.logger.info('train set convert finish')
        self.npz2tfrecord(target_path, test_tfrecord, test_set)
        log.logger.info('test set convert finish')


if __name__ == '__main__':
    splitor = NpzDataSplitor(10)
    splitor.random_split_of_per_user(r'D:\实验数据\2021\毕设\micarrayspeaker\npz',
                                     r'D:\实验数据\2021\毕设\micarrayspeaker\tfrecord',
                                     'train.tfrecord',
                                     'test.tfrecord')
