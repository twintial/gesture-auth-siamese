import tensorflow as tf
import numpy as np
from scipy.signal import butter, filtfilt
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from config import *
def butter_bandpass_filter(data, lowcut, highcut, fs=48e3, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, data)

def get_cos_IQ_raw(data: np.ndarray, f, fs=48e3) -> (np.ndarray, np.ndarray):
    frames = data.shape[1]
    offset = 0
    times = np.arange(offset, offset+frames) * 1 / fs
    I_raw = np.cos(2 * np.pi * f * times) * data
    Q_raw = -np.sin(2 * np.pi * f * times) * data
    return I_raw, Q_raw

def move_average_overlap_filter(data, win_size=200, overlap=100, axis=-1):
    if len(data.shape) == 1:
        data = data.reshape((1, -1))
    ret = np.cumsum(data, axis=axis)
    ret[:, win_size:] = ret[:, win_size:] - ret[:, :-win_size]
    result = ret[:, win_size - 1:] / win_size
    index = np.arange(0, result.shape[1], overlap)
    return result[:, index]

def get_phase(I: np.ndarray, Q: np.ndarray) -> np.ndarray:
    signal = I + 1j * Q
    angle = np.angle(signal)
    unwrap_angle = np.unwrap(angle)
    return unwrap_angle

def get_magnitude(I: np.ndarray, Q: np.ndarray) -> np.ndarray:
    signal = I + 1j * Q
    magn = np.abs(signal)
    magn = 10*np.log10(magn)
    return magn

# 生成tfrecord
def tensor_feature(tensor):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tensor).numpy()]))
def create_example(filename):
    audio_binary = tf.io.read_file(filename)
    data, fs = tf.audio.decode_wav(audio_binary)  # 会变成-1，1
    data = data.numpy().T[:-1, int(fs * DELAY_TIME):]
    # 开始处理数据
    unwrapped_phase_list = []
    unwrapped_phase_diff_list = []
    magnitude_list = []
    for i in range(NUM_OF_FREQ):
        fc = F0 + i * STEP
        data_filter = butter_bandpass_filter(data, fc - 150.0, fc + 150.0)
        I_raw, Q_raw = get_cos_IQ_raw(data_filter, fc, fs)
        # 滤波+下采样
        I = move_average_overlap_filter(I_raw)
        Q = move_average_overlap_filter(Q_raw)

        unwrapped_phase = get_phase(I, Q)
        unwrapped_phase_list.append(unwrapped_phase)
        unwrapped_phase_diff_list.append(np.diff(unwrapped_phase))
        magnitude = get_magnitude(I, Q)
        magnitude_list.append(magnitude)
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'unwrapped_phase_list': tensor_feature(tf.constant(unwrapped_phase_list, dtype=tf.float32)),
                'unwrapped_phase_diff_list': tensor_feature(tf.constant(unwrapped_phase_diff_list, dtype=tf.float32)),
                'magnitude_list': tensor_feature(tf.constant(magnitude_list, dtype=tf.float32)),
            }
        ))
    return tf_example
def generate_tfrecord():
    with tf.io.TFRecordWriter(r'dataset/test.tfrecord') as f:
        for dir in TRAINING_AUDIO_DIRS:
            filenames = os.listdir(dir)
            for filename in filenames:
                audio_file = os.path.join(dir, filename)
                print(audio_file)
                example = create_example(audio_file)
                f.write(example.SerializeToString())
            print(f'complete {dir}')


# 解析tfrecord
def parse_example(example_proto):
    feature_description = {
        'unwrapped_phase_list': tf.io.VarLenFeature(tf.string),
        'unwrapped_phase_diff_list': tf.io.VarLenFeature(tf.string),
        'magnitude_list': tf.io.VarLenFeature(tf.string),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    parsed_example['unwrapped_phase_list'] = tf.io.parse_tensor(parsed_example['unwrapped_phase_list'].values[0], tf.float32)
    parsed_example['unwrapped_phase_diff_list'] = tf.io.parse_tensor(parsed_example['unwrapped_phase_diff_list'].values[0], tf.float32)
    parsed_example['magnitude_list'] = tf.io.parse_tensor(parsed_example['magnitude_list'].values[0], tf.float32)
    return parsed_example
def get_data_from_tfrecord():
    train_dataset = tf.data.TFRecordDataset([r'dataset/test.tfrecord'])
    train_dataset = train_dataset.map(parse_example)
    train_dataset.as_numpy_iterator()
    train_dataset.enumerate()
    for d in train_dataset:
        print(d)



if __name__ == '__main__':
    # generate_tfrecord()
    get_data_from_tfrecord()
