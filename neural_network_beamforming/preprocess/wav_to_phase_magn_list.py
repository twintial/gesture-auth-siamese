import numpy as np
from scipy.signal import butter, filtfilt

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


def padding_or_clip(array: np.ndarray, target_len):
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
def convert_wavfile_to_IQ(filename):
    # # 这个可能耗时有点长
    # audio_binary = tf.io.read_file(filename)
    # data, fs = tf.audio.decode_wav(audio_binary)  # 会变成-t，t
    # data = data.numpy().T[:-t, int(fs * DELAY_TIME):]
    data, fs = load_audio_data(filename, 'wav')
    assert fs == FS
    data = data.T[:-1, int(fs * DELAY_TIME):]
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
        unwrapped_phase_diff_padded = padding_or_clip(unwrapped_phase_diff, PADDING_LEN)
        magnitude_diff_padded = padding_or_clip(magnitude_diff, PADDING_LEN)

        # plt.figure()
        # plt.plot(unwrapped_phase_diff[0])
        # plt.figure()
        # plt.plot(unwrapped_phase_diff_padded[0])
        # plt.figure()
        # plt.plot(magnitude_diff[0])
        # plt.figure()
        # plt.plot(magnitude_diff_padded[0])
        # plt.show()

        unwrapped_phase_diff_list.append(unwrapped_phase_diff_padded)
        magnitude_diff_list.append(magnitude_diff_padded)

    return np.array(unwrapped_phase_diff_list).reshape(data_shape), np.array(magnitude_diff_list).reshape(data_shape)
