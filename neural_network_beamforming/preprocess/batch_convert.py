import os
import re
import numpy as np
import log
from neural_network_beamforming.preprocess.wav_to_iq_list import convert_wavfile_to_IQ


def convert(raw_audio_dir, target_npz_dir):
    user_dirs = os.listdir(raw_audio_dir)
    for user_dir in user_dirs:
        gesture_dirs = os.listdir(os.path.join(raw_audio_dir, user_dir))
        for gesture_dir in gesture_dirs:
            abs_gesture_dir = os.path.join(raw_audio_dir, user_dir, gesture_dir)
            audio_files = os.listdir(abs_gesture_dir)
            save_dir = os.path.join(target_npz_dir, user_dir, gesture_dir)
            os.makedirs(save_dir, exist_ok=False)
            for audio_file in audio_files:
                m = re.match(r'(\d*)\.wav', audio_file)
                if m:
                    abs_audio_file = os.path.join(abs_gesture_dir, audio_file)
                    I_list, Q_list = convert_wavfile_to_IQ(abs_audio_file)
                    save_file = os.path.join(save_dir, m.group(1))
                    np.savez_compressed(save_file,
                                        I_list=I_list,
                                        Q_list=Q_list)
                else:
                    raise IOError(f'{audio_file} in {os.path.join(raw_audio_dir, user_dir)} is not a wav file')
        log.logger.debug(f'convert wav files in {os.path.join(raw_audio_dir, user_dir)} completely!')


if __name__ == '__main__':
    # train
    # convert(r'D:\实验数据\2021\siamese\train', r'D:\实验数据\2021\siamese\train_npz')
    # test_demo
    # convert(r'D:\实验数据\2021\siamese\test', r'D:\实验数据\2021\siamese\test_npz')
    # convert(r'D:\实验数据\2021\毕设\micarrayspeaker\raw', r'D:\实验数据\2021\毕设\micarrayspeaker\npz')
    convert(r'D:\实验数据\2021\neuralbeamform\audataset\raw', r'D:\实验数据\2021\neuralbeamform\audataset\npz')