import os
import numpy as np
import re

from scipy.signal import butter, filtfilt
from concurrent.futures import ThreadPoolExecutor

import log
from config import *
from siamese_cons_loss.preprocess.util import load_audio_data
def butter_bandpass_filter(data, lowcut, highcut, fs=48e3, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, data)


def get_cos_IQ_raw(data: np.ndarray, f, fs=48e3) -> (np.ndarray, np.ndarray):
    frames = data.shape[1]
    offset = 0
    times = np.arange(offset, offset + frames) * 1 / fs
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
    magn = 10 * np.log10(magn)
    return magn


def zero_padding_or_clip(array: np.ndarray, target_len):
    array_len = array.shape[1]
    delta_len = array_len - target_len
    if delta_len > 0:
        left_clip_len = abs(delta_len) // 2
        right_clip_len = abs(delta_len) - left_clip_len
        return array[:, left_clip_len:-right_clip_len]
    elif delta_len < 0:
        left_zero_padding_len = abs(delta_len) // 2
        right_zero_padding_len = abs(delta_len) - left_zero_padding_len
        return np.pad(array, ((0, 0), (left_zero_padding_len, right_zero_padding_len)))

# main function
def convert_wavfile_to_phase_and_magnitude(data, fs):
    # 开始处理数据
    unwrapped_phase_diff_list = []
    magnitude_diff_list = []
    for i in range(NUM_OF_FREQ):
        fc = F0 + i * STEP
        data_filter = butter_bandpass_filter(data, fc - 150.0, fc + 150.0, fs)
        I_raw, Q_raw = get_cos_IQ_raw(data_filter, fc, fs)
        # 滤波+下采样
        I = move_average_overlap_filter(I_raw[:, I_Q_skip:-I_Q_skip])
        Q = move_average_overlap_filter(Q_raw[:, I_Q_skip:-I_Q_skip])

        # 暂时不做平滑

        unwrapped_phase = get_phase(I, Q)
        unwrapped_phase_diff = np.diff(unwrapped_phase)
        magnitude = get_magnitude(I, Q)
        magnitude_diff = np.diff(magnitude)
        # padding，是不是可以放到外面做
        unwrapped_phase_diff_padded = zero_padding_or_clip(unwrapped_phase_diff, PADDING_LEN)
        magnitude_diff_padded = zero_padding_or_clip(magnitude_diff, PADDING_LEN)

        unwrapped_phase_diff_list.append(unwrapped_phase_diff_padded)
        magnitude_diff_list.append(magnitude_diff_padded)

    return np.array(unwrapped_phase_diff_list).reshape(data_shape), np.array(magnitude_diff_list).reshape(data_shape)


def augmentation_op(abs_audio_file, augmentation_save_files, filename):
    data, fs = load_audio_data(abs_audio_file, 'wav')
    assert fs == FS
    data = data.T[:-1, int(fs * DELAY_TIME):]
    # augment
    for i in range(5):
        # rotation
        data_i = np.vstack((data[0], data[i + 2:], data[1:i + 2]))
        save_dir = augmentation_save_files[i]
        phase_diff, magn_diff = convert_wavfile_to_phase_and_magnitude(data_i, fs)
        save_file = os.path.join(save_dir, filename)
        np.savez_compressed(save_file,
                            phase_diff=phase_diff,
                            magn_diff=magn_diff)

def rotation_augmentation(raw_audio_dir, target_npz_dir):
    user_dirs = os.listdir(raw_audio_dir)
    # 好像有点问题
    pool = ThreadPoolExecutor(max_workers=4)
    for user_dir in user_dirs:
        gesture_dirs = os.listdir(os.path.join(raw_audio_dir, user_dir))
        for gesture_dir in gesture_dirs:
            abs_gesture_dir = os.path.join(raw_audio_dir, user_dir, gesture_dir)
            audio_files = os.listdir(abs_gesture_dir)
            augmentation_save_files = []
            for i in range(5):
                augmentation_save_files.append(os.path.join(target_npz_dir, user_dir, f'{gesture_dir}_au{i+1}'))
                os.makedirs(augmentation_save_files[i], exist_ok=False)
            for audio_file in audio_files:
                m = re.match(r'(\d*)\.wav', audio_file)
                if m:
                    abs_audio_file = os.path.join(abs_gesture_dir, audio_file)
                    # augmentation_op(abs_audio_file, augmentation_save_files, m.group(1))
                    pool.submit(augmentation_op, abs_audio_file, augmentation_save_files, m.group(1))
                else:
                    raise IOError(f'{audio_file} in {os.path.join(raw_audio_dir, user_dir)} is not a wav file')
    pool.shutdown()
                        

if __name__ == '__main__':
    rotation_augmentation(r'D:\实验数据\2021\毕设\micarrayspeaker\dataset1\raw', r'D:\实验数据\2021\毕设\micarrayspeaker\audataset\npz')