import os
import numpy as np
import tensorflow as tf
import log

def tensor_feature(tensor):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tensor).numpy()]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class PhasePairConvertor:
    def __init__(self, npz_path, target_path):
        self.npz_path = npz_path
        self.target_path = target_path
        self.path_label_tuples = self._create_path_label_tuples()


    def _create_path_label_tuples(self):
        dir_list = []
        user_dirs = os.listdir(self.npz_path)
        for user_dir in user_dirs:
            gesture_dirs = os.listdir(os.path.join(self.npz_path, user_dir))
            for gesture_dir in gesture_dirs:
                abs_gesture_dir = os.path.join(self.npz_path, user_dir, gesture_dir)
                dir_list.append(abs_gesture_dir)
        labels = np.arange(len(dir_list))
        return list(zip(dir_list, labels))


    def create_pairs(self):
        with tf.io.TFRecordWriter(self.target_path) as f:
            # create positive pairs
            log.logger.info('start to create positive pairs')
            for path_label_tuple in self.path_label_tuples:
                npz_file_names = os.listdir(path_label_tuple[0])
                file_num = len(npz_file_names)
                for i in range(file_num):
                    data_i = np.load(os.path.join(path_label_tuple[0], npz_file_names[i]))
                    phase_diff_i = data_i['phase_diff']
                    for j in range(i + 1, file_num):
                        data_j = np.load(os.path.join(path_label_tuple[0], npz_file_names[j]))
                        phase_diff_j = data_j['phase_diff']
                        # need
                        positive_pair = [phase_diff_i, phase_diff_j]
                        label = 1
                        tf_example = tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    'pair': tensor_feature(tf.constant(positive_pair, dtype=tf.float32)),
                                    'label': int64_feature(label)
                                }
                            ))
                        f.write(tf_example.SerializeToString())
                log.logger.info(f'finish {path_label_tuple[0]}')
            # create negative pairs
            log.logger.info('start to create negative pairs')
            tuple_num = len(self.path_label_tuples)
            for i in range(tuple_num):
                pass




if __name__ == '__main__':
    convertor = PhasePairConvertor(r'D:\实验数据\2021\siamese\train_npz', r'D:\实验数据\2021\siamese\train_tfrecord\test.tfrecord')
    convertor.create_pairs()

