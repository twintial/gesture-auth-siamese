import os
import numpy as np

import log


def rotation_augmentation(raw_audio_dir, target_npz_dir):
    user_dirs = os.listdir(raw_audio_dir)
    for user_dir in user_dirs:
        gesture_dirs = os.listdir(os.path.join(raw_audio_dir, user_dir))
        for gesture_dir in gesture_dirs:
            abs_gesture_dir = os.path.join(raw_audio_dir, user_dir, gesture_dir)
            audio_files = os.listdir(abs_gesture_dir)

            # 这里开始改
