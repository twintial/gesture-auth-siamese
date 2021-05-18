import os
import numpy as np
import tensorflow as tf


def tensor_feature(tensor):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tensor).numpy()]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# r'/media/home/shenjj/dataset/siamese/newgesture/npz'
def generate(data_paths, target_path, tfrecord_file_name):
    p_num = 0
    n_num = 0
    npz_phases_num_dict = {}
    for data_path in data_paths:
        people_names = os.listdir(data_path)
        for people_name in people_names:
            gesture_codes = os.listdir(os.path.join(data_path, people_name))
            for gesture_code in gesture_codes:
                npz_phases = os.listdir(os.path.join(data_path, people_name, gesture_code))
                npz_phases_num_dict[os.path.join(data_path, people_name, gesture_code)] = len(
                    npz_phases)
    with tf.io.TFRecordWriter(os.path.join(target_path, tfrecord_file_name)) as f:
        # 正样本对
        for path, num in npz_phases_num_dict.items():
            npz_phases = os.listdir(path)
            # random shuffle
            np.random.shuffle(npz_phases)
            anchor_file = npz_phases[0]
            positive_file = npz_phases[1]

            data_a = np.load(os.path.join(path, anchor_file))
            phase_diff_a: np.ndarray = data_a['phase_diff']

            data_p = np.load(os.path.join(path, positive_file))
            phase_diff_p: np.ndarray = data_p['phase_diff']

            positive_pair = [phase_diff_a, phase_diff_p]
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
            p_num += 1
        # 负样本对
        for i in range(5000):
            npz_phases_paths = list(npz_phases_num_dict.keys())
            # random shuffle
            np.random.shuffle(npz_phases_paths)
            # 正样本path
            path_a = npz_phases_paths[0]
            # 负样本path
            path_n = npz_phases_paths[1]
            npz_phases_a = os.listdir(path_a)
            np.random.shuffle(npz_phases_a)
            anchor_file = npz_phases_a[0]

            npz_phases_n = os.listdir(path_n)
            np.random.shuffle(npz_phases_n)
            negative_file = npz_phases_n[0]

            data_a = np.load(os.path.join(path_a, anchor_file))
            phase_diff_a: np.ndarray = data_a['phase_diff']

            data_n = np.load(os.path.join(path_n, negative_file))
            phase_diff_n: np.ndarray = data_n['phase_diff']

            negative_pair = [phase_diff_a, phase_diff_n]
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
            n_num += 1

    print(p_num)
    print(n_num)


if __name__ == '__main__':
    generate([r'/media/home/shenjj/dataset/siamese/newgesture/npz'], r'/media/home/shenjj/dataset/siamese/newgesture', 'test.tfrecord')