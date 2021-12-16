import torch.utils.data
import torch
import numpy as np
import h5py
from tqdm import tqdm
from torch.autograd import Variable
import os
import pickle


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, LABEL_TO_INDEX, time_downsample_factor=1, num_channel=9, loadAllinMem=True):
        '''

        :param path: Path of dataset
        :param time_downsample_factor:
        :param num_channel: 4 = RGB+NIR
        :param loadAllinMem: True to load all data in memory once (accelerate I/O time)
        '''
        self.num_channel = num_channel
        self.time_downsample_factor = time_downsample_factor
        self.eval_mode = False
        self.path = path
        self.time_downsample_factor = time_downsample_factor
        self.loaAllinMem = loadAllinMem
        self.load_data(loadAllinMem)
        self.label_to_index = LABEL_TO_INDEX

    def load_data(self, loadAllinMem):
        print("Start loading...")
        if loadAllinMem:
            # Load all data in memory
            with h5py.File(self.path, "r", libver='latest', swmr=True) as f:
                self.data = f["data"][:].copy()
                self.gt = f["gt"][:].copy()
            temporal_length = self.data.shape[-2]

        else:
            # Open the data file
            f = h5py.File(self.path, "r", libver='latest', swmr=True)
            self.data = f["data"]
            self.gt = f["gt"]

        data_shape = self.data.shape
        target_shape = self.gt.shape
        self.num_samples = data_shape[0]

        if len(target_shape) == 3:
            self.eval_mode = True
            self.num_pixels = target_shape[0] * target_shape[1] * target_shape[2]
        else:
            self.num_pixels = target_shape[0]

        label_idxs = np.unique(self.gt)
        self.n_classes = len(label_idxs)
        self.temporal_length = data_shape[-2] // self.time_downsample_factor

        print('Number of pixels: ', self.num_pixels)
        print('Number of classes: ', self.n_classes)
        print('Temporal length: ', self.temporal_length)
        print('Number of channels: ', self.num_channel)

    def return_labels(self):
        return self.gt

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Convert numpy array to torch tensor
        X = torch.from_numpy(self.data[idx])
        # Map target original label to index
        target = self.gt[idx]
        mapping_target = np.zeros_like(target)
        for key in self.label_to_index:
            mapping_target[target == key] = self.label_to_index[key]

        mapping_target = torch.from_numpy(np.array(mapping_target)).float()
        # Temporal down-sampling
        X = X[..., 0::self.time_downsample_factor, :self.num_channel]

        # keep values between 0-1
        X = X * 1e-4

        return X.float(), mapping_target.long()



if __name__=="__main__":
    dset_type = "test"
    data_path = f"D:\jingyli\II_Lab3\data\imgint_{dset_type}set_v2.hdf5"
    testset = Dataset(data_path, loadAllinMem=True, num_channel=4,
                      LABEL_TO_INDEX={0: -1,
                            21: 0, 51: 1, 20: 2, 27: 3, 38: 4,
                            49: 5, 50: 6, 45: 7, 30: 8, 48: 9,
                            42: 10, 46: 11, 36: 12})
    X, y = testset.__getitem__(0)
    print(X.shape)
    print(y.shape)