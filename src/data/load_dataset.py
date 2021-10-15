import torch.utils.data as td
import torch
import h5py

class Waveform_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, file_name=('X.hdf5'), seed=0, size=None):
        self._init_dataset(dataset_path, file_name)
        
        # allow limitation of max samples
        if size is not None and size < self.size:
            self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])

    def _init_dataset(self, dataset_path, file_name):
        # features file
        with h5py.File(dataset_path / file_name, 'r') as X_file:
            self.X = X_file['sinusoids'][:]
            self.size = len(self.X)