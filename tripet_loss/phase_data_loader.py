import os
import numpy as np


class PhaseDataLoader:
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.npz_phases_num_dict = {}
        self._get_npz_paths_num()

    def _get_npz_paths_num(self):
        for data_path in self.data_paths:
            people_names = os.listdir(data_path)
            for people_name in people_names:
                gesture_codes = os.listdir(os.path.join(data_path, people_name))
                for gesture_code in gesture_codes:
                    npz_phases = os.listdir(os.path.join(data_path, people_name, gesture_code))
                    self.npz_phases_num_dict[os.path.join(data_path, people_name, gesture_code)] = len(
                        npz_phases)

    def get_random_batch(self, nof_class_per_batch, nof_phases_per_class):
        train_batch = []
        npz_phases_paths = list(self.npz_phases_num_dict.keys())
        # random
        np.random.shuffle(npz_phases_paths)
        random_npz_phases_paths = npz_phases_paths[:nof_class_per_batch]
        for random_npz_phases_path in random_npz_phases_paths:
            npz_phases = os.listdir(random_npz_phases_path)
            # random
            np.random.shuffle(npz_phases)
            npz_phases = npz_phases[:nof_phases_per_class]
            for npz_phase in npz_phases:
                data = np.load(os.path.join(random_npz_phases_path, npz_phase))
                phase_diff: np.ndarray = data['phase_diff']
                train_batch.append(phase_diff[..., np.newaxis])
        return np.array(train_batch)