import tensorflow as tf


class PhasePairLoader:
    def __init__(self, train_set_files, test_set_files, batch_size):
        self.train_set_files = train_set_files
        self.test_set_files = test_set_files
        self.batch_size = batch_size

        self.train_set = self.load_train_set()
        self.test_set = self.load_test_set()

    def load_train_set(self):
        train_dataset = tf.data.TFRecordDataset(self.train_set_files)
        return train_dataset.batch(self.batch_size).prefetch(1)

    def load_test_set(self):
        test_dataset = tf.data.TFRecordDataset(self.test_set_files)
        return test_dataset.batch(self.batch_size).prefetch(1)

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set
