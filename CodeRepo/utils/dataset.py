import torch.utils.data
import torch
import numpy as np
import h5py
import pickle


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, LABEL_TO_INDEX, time_downsample_factor=1, num_channel=9, loadAllinMem=True, dset_type="train"):
        '''

        :param path:
        :param LABEL_TO_INDEX: Map labels (0-51) to index (-1 - 12)
        :param time_downsample_factor:
        :param num_channel:
        :param loadAllinMem: True to load all data in memory once (accelerate I/O time)
        :param dset_type:
        '''
        self.num_channel = num_channel
        self.time_downsample_factor = time_downsample_factor
        self.eval_mode = False
        self.path = path
        self.time_downsample_factor = time_downsample_factor
        self.loaAllinMem = loadAllinMem
        self.dset_type = dset_type
        self.load_data(loadAllinMem)
        self.label_to_index = LABEL_TO_INDEX

    def load_data(self, loadAllinMem):
        print("Start loading...")
        if loadAllinMem:
            # Load all data in memory
            with h5py.File(self.path, "r", libver='latest', swmr=True) as f:
                self.data = f["data"][:].copy()
                self.gt = f["gt"][:].copy()
            # Reshape test data from (24,24,71,4) to (-1,71,4)
            temporal_length = self.data.shape[-2]
            if self.dset_type == "test":
                self.data = self.data.reshape(-1, temporal_length, self.num_channel)
                self.gt = self.gt.reshape(-1)
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
        X = self.data[idx]
        target = self.gt[idx]

        # Convert numpy array to torch tensor
        X = torch.from_numpy(X)
        # TODO Map target original label to index
        target = self.label_to_index[target]
        target = torch.from_numpy(np.array(target)).float()

        # Temporal down-sampling
        X = X[...,0::self.time_downsample_factor, :self.num_channel]

        # keep values between 0-1
        X = X * 1e-4

        return X.float(), target.long()



import matplotlib.pyplot as plt

colordict = {'B04': '#a6cee3', 'NDWI': '#1f78b4', 'NDVI': '#b2df8a', 'RATIOVVVH': '#33a02c', 'B09': '#fb9a99',
             'B8A': '#e31a1c', 'IRECI': '#fdbf6f', 'B07': '#ff7f00', 'B12': '#cab2d6', 'B02': '#6a3d9a', 'B03': '#0f1b5f',
             'B01': '#b15928', 'B10': '#005293', 'VH': '#98c6ea', 'B08': '#e37222', 'VV': '#a2ad00', 'B05': '#69085a',
             'B11': '#007c30', 'NDVVVH': '#00778a', 'BRIGHTNESS': '#000000', 'B06': '#0f1b5f'}
plotbands = ["B02", "B03", "B04", "B08"]

label_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
          26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51]

label_names = ['Unknown', 'Apples', 'Beets', 'Berries', 'Biodiversity area', 'Buckwheat',
               'Chestnut', 'Chicory', 'Einkorn wheat', 'Fallow', 'Field bean', 'Forest',
               'Gardens', 'Grain', 'Hedge', 'Hemp', 'Hops', 'Legumes', 'Linen', 'Lupine',
               'Maize', 'Meadow', 'Mixed crop', 'Multiple', 'Mustard', 'Oat', 'Pasture', 'Pears',
               'Peas', 'Potatoes', 'Pumpkin', 'Rye', 'Sorghum', 'Soy', 'Spelt', 'Stone fruit',
               'Sugar beet', 'Summer barley', 'Summer rapeseed', 'Summer wheat', 'Sunflowers',
               'Tobacco', 'Tree crop', 'Vegetables', 'Vines', 'Wheat', 'Winter barley',
               'Winter rapeseed', 'Winter wheat']

def plot_bands(X):
    x = np.arange(X.shape[0])
    for i, band in enumerate(plotbands):
        plt.plot(x, X[:,i])

    plt.savefig("bands.png", dpi=300, format="png", bbox_inches='tight')

if __name__ == "__main__":
    dset_type = "train"
    data_path = f"D:\jingyli\TCN-CropClassification\data\imgint_{dset_type}set_v2.hdf5"
    traindataset = Dataset(data_path, loadAllinMem=True, dset_type=dset_type)
    X,y = traindataset.__getitem__(0)
    print(X.shape)
    print(y.shape)
    gt_list = traindataset.return_labels()
    labels, pix_counts = np.unique(gt_list, return_counts=True)
    print('Labels: ', labels)

    # Plot histogram
    inds = pix_counts.argsort()
    pix_counts_sorted = pix_counts[inds]
    labels_sorted = labels[inds]
    print(labels_sorted)

    label_names_sorted = [label_names[label_index.index(x)] for x in labels_sorted]
    label_idx_sorted = labels_sorted
    print(label_names_sorted[::-1])
    print(label_idx_sorted[::-1])
    label_to_idx = {l: _ for _, l in enumerate(label_idx_sorted[::-1])}
    print(label_to_idx)

    fig = plt.figure()
    plt.bar(label_names_sorted, pix_counts_sorted)
    plt.xticks(rotation=90)
    plt.savefig(f"hist_{dset_type}.png", dpi=300, format="png", bbox_inches='tight')

    # Mapping index with labels
    print("Pixel counts, ", pix_counts_sorted[::-1])
    with open(f"label_count_{dset_type}.pkl", "wb") as f:
        pickle.dump(pix_counts_sorted[::-1], f)

