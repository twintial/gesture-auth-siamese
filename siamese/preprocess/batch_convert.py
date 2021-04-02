import os
import re
from siamese.preprocess.wav_to_phase_magn_list import convert_wavfile_to_phase_and_magnitude
import numpy as np


def convert(raw_audio_dir, target_npz_dir):
    user_dirs = os.listdir(raw_audio_dir)
    for user_dir in user_dirs:
        gesture_dirs = os.listdir(os.path.join(raw_audio_dir, user_dir))
        for gesture_dir in gesture_dirs:
            abs_gesture_dir = os.path.join(raw_audio_dir, user_dir, gesture_dir)
            audio_files = os.listdir(abs_gesture_dir)
            save_dir = os.path.join(target_npz_dir, user_dir, gesture_dir)
            os.makedirs(save_dir, exist_ok=True)
            for audio_file in audio_files:
                m = re.match(r'(\d*)\.wav', audio_file)
                if m:
                    abs_audio_file = os.path.join(abs_gesture_dir, audio_file)
                    phase_list, magn_list = convert_wavfile_to_phase_and_magnitude(abs_audio_file)
                    save_file = os.path.join(save_dir, m.group(1))
                    np.savez_compressed(save_file,
                                        phase_list=phase_list,
                                        magn_list=magn_list)
                else:
                    raise IOError('not a wav file')


if __name__ == '__main__':
    convert(r'D:\实验数据\2021\siamese\train', r'D:\实验数据\2021\siamese\train_npz')
