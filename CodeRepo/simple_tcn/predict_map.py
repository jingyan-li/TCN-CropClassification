# Validate test dataset AND SAVE IN AS A H5PY MAP
from tqdm import tqdm
import pickle
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import sys
import pandas as pd
import h5py
from utils.metrics import get_label_weights, get_confmat_metrics

sys.path.append("../../")
from simple_tcn.model import TCN
from simple_tcn.test_config import config
from utils.testset import Dataset


checkpoint_name = 'last.pt'

# Read configs
SEED = config["seed"]
CUDA = config["cuda"]
TEST = config["test"]
batch_size = 8  # config["batch_size"]
kernel_size = config["kernel-size"]
dropout = config["dropout"]
n_hunits = config["nhid"]
n_levels = config["levels"]
checkpoint_path = config["checkpoint-path"]
result_csv_path = os.path.join(checkpoint_path, checkpoint_name[:-3]+"_test_metrics.csv")
result_conf_path = os.path.join(checkpoint_path, checkpoint_name[:-3]+"_test_confmat.csv")
LABEL_TO_INDEX = config["label-to-index"]
label_names = config["label-names"]


n_classes = 13
input_channels = 4
output_channels = [n_hunits]*n_levels
# Model
model = TCN(input_channels, n_classes, output_channels, kernel_size=kernel_size, dropout=dropout)
if CUDA:
    model.cuda()
# Load checkpoint
checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_name))
model.load_state_dict(checkpoint['model_state_dict'])


test_dset = Dataset(path="../../data/imgint_testset_v2.hdf5",
                    LABEL_TO_INDEX= LABEL_TO_INDEX,
                        time_downsample_factor=1,
                        num_channel=4)

test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, num_workers=0, shuffle=False)


def predict_map(pred_path,tf_path):
    correct = 0
    target_num = 0
    with torch.no_grad():
        # confusion_matrix = torch.zeros(n_classes, n_classes)
        vis_pred = torch.empty(size=(0, 24, 24))
        vis_tf = torch.empty(size=(0, 24, 24))
        vis_pred = vis_pred.cuda()
        vis_tf = vis_tf.cuda()
        for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
            data, target = data.cuda(), target.cuda()
            # print(data, target)
            data = data.reshape(-1, data.shape[-2], data.shape[-1])
            target = target.reshape(-1, 1)
            # if target.unique().size()[0] >1:
            #     print("Has DATA")
            data = torch.swapaxes(data, 1, 2)
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=False)[1]
            pred = pred.reshape(-1, 1)
            pred[target == -1] = 99  # Undefined pixels are marked as 99
            tf = target == pred
            tf = tf.int()*100   # Correct pred: 100; Wrong pred: 0
            tf[target == -1] = 99  # Undefined pixels are marked as 99
            pred = pred.reshape(-1, 24, 24)
            target = target.reshape(-1, 24, 24)
            tf = tf.reshape(-1, 24, 24)
            vis_pred = torch.cat((vis_pred, pred), 0)
            vis_tf = torch.cat((vis_tf, tf), 0)
        vis_tf = vis_tf.cpu()
        vis_pred = vis_pred.cpu()
        print(vis_pred.shape)
        hf_vis = h5py.File(pred_path, "w")
        hf_vis.create_dataset('pred', data=vis_pred)
        hf_vis_tf = h5py.File(tf_path, "w")
        hf_vis_tf.create_dataset('tf', data=vis_tf)
        print("finished")



if __name__ == "__main__":
    pred_path = os.path.join(checkpoint_path, checkpoint_name[:-3]+"_predict.h5")
    tf_path = os.path.join(checkpoint_path, checkpoint_name[:-3]+"_tf.h5")
    predict_map(pred_path,tf_path)