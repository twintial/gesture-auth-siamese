import tensorflow as tf


def parse_example(example_proto):
    feature_description = {
        'pair': tf.io.VarLenFeature(tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    parsed_example['pair'] = tf.io.parse_tensor(parsed_example['pair'].values[0], tf.float32)
    label = parsed_example['label']
    return parsed_example

class PhasePairLoader:
    def __init__(self, train_set_files, test_set_files, batch_size):
        self.train_set_files = train_set_files
        self.test_set_files = test_set_files
        self.batch_size = batch_size

        self.train_set = self._load_train_set()
        self.test_set = self._load_test_set()

    @staticmethod
    def _parse_example(example_proto):
        feature_description = {
            'pair': tf.io.VarLenFeature(tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        pair = tf.io.parse_tensor(parsed_example['pair'].values[0], tf.float32)[..., tf.newaxis]
        label = parsed_example['label']
        return pair, label

    def _load_train_set(self):
        train_set = tf.data.TFRecordDataset(self.train_set_files)
        train_set = train_set.map(self._parse_example)
        return train_set.batch(self.batch_size).prefetch(1)

    def _load_test_set(self):
        test_set = tf.data.TFRecordDataset(self.test_set_files)
        test_set = test_set.map(self._parse_example)
        return test_set.batch(self.batch_size).prefetch(1)

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set


if __name__ == '__main__':
    # test
    loader = PhasePairLoader([r'D:\实验数据\2021\siamese\train_tfrecord\train.tfrecord'],
                             [r'D:\实验数据\2021\siamese\test_tfrecord\test.tfrecord'], 10)
    t = loader.get_train_set()
    for d in t:
        print(d)
