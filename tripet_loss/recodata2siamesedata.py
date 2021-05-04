import os
import re
import shutil


def reco2siamese(reco_path, siamese_path):
    persons = os.listdir(reco_path)
    for person in persons:
        gesture_dirs = os.listdir(os.path.join(reco_path, person))
        for gesture_dir in gesture_dirs:
            if 'au' in gesture_dir:
                continue
            gestures = os.listdir(os.path.join(reco_path, person, gesture_dir))
            for gesture in gestures:
                m = re.match(r'(\d*)\.npz', gesture)
                if m:
                    code = m.group(1)
                    label = int(code) // 10
                    os.makedirs(os.path.join(siamese_path, person, str(label)), exist_ok=True)
                    src = os.path.join(reco_path, person, gesture_dir, gesture)
                    dist = os.path.join(siamese_path, person, str(label), f'{gesture_dir}_{gesture}')
                    shutil.copy(src, dist)



if __name__ == '__main__':
    reco2siamese(r'D:\实验数据\2021\newgesture\npz', r'D:\实验数据\2021\siamese\newgesture\npz')