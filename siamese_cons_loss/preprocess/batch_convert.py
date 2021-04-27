import os
import re
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import log
from siamese_cons_loss.preprocess.wav_to_phase_magn_list import convert_wavfile_to_phase_and_magnitude


def do_save(abs_audio_file, save_dir, m):
    phase_diff, magn_diff = convert_wavfile_to_phase_and_magnitude(abs_audio_file)
    save_file = os.path.join(save_dir, m.group(1))
    np.savez_compressed(save_file,
                        phase_diff=phase_diff,
                        magn_diff=magn_diff)


def convert(raw_audio_dir, target_npz_dir):
    user_dirs = os.listdir(raw_audio_dir)
    pool = ThreadPoolExecutor(max_workers=4)
    for user_dir in user_dirs:
        gesture_dirs = os.listdir(os.path.join(raw_audio_dir, user_dir))
        for gesture_dir in gesture_dirs:
            abs_gesture_dir = os.path.join(raw_audio_dir, user_dir, gesture_dir)
            audio_files = os.listdir(abs_gesture_dir)
            save_dir = os.path.join(target_npz_dir, user_dir, gesture_dir)
            try:
                os.makedirs(save_dir, exist_ok=False)
            except FileExistsError:
                log.logger.warning(f'{save_dir} exists, fail to create')
                continue
            for audio_file in audio_files:
                m = re.match(r'(\d*)\.wav', audio_file)
                if m:
                    abs_audio_file = os.path.join(abs_gesture_dir, audio_file)
                    pool.submit(do_save, abs_audio_file, save_dir, m)
                    # phase_diff, magn_diff = convert_wavfile_to_phase_and_magnitude(abs_audio_file)
                    # save_file = os.path.join(save_dir, m.group(1))
                    # np.savez_compressed(save_file,
                    #                     phase_diff=phase_diff,
                    #                     magn_diff=magn_diff)
                else:
                    raise IOError(f'{audio_file} in {os.path.join(raw_audio_dir, user_dir)} is not a wav file')
        log.logger.debug(f'convert wav files in {os.path.join(raw_audio_dir, user_dir)} completely!')
    pool.shutdown()

if __name__ == '__main__':
    # train
    # convert(r'D:\实验数据\2021\siamese\train', r'D:\实验数据\2021\siamese\train_npz')
    # test_demo
    # convert(r'D:\实验数据\2021\siamese\test', r'D:\实验数据\2021\siamese\test_npz')
    convert(r'D:\实验数据\2021\newgesture\raw', r'D:\实验数据\2021\newgesture\npz')
