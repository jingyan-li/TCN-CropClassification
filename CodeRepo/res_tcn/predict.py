# Validate test dataset
from tqdm import tqdm
import pickle
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import sys
import pandas as pd

from utils.metrics import get_label_weights, get_confmat_metrics

sys.path.append("../../")
from res_tcn.model import TCN
from res_tcn.test_config import config
from utils.dataset import Dataset


checkpoint_name = 'last.pt'

# Read configs
SEED = config["seed"]
CUDA = config["cuda"]
TEST = config["test"]
batch_size = config["batch_size"]
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
                        num_channel=4, dset_type="test")

test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, num_workers=0, shuffle=False)


def make_predict():
    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        confusion_matrix = torch.zeros(n_classes, n_classes)
        for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
            if CUDA:
                data, target = data.cuda(), target.cuda()
            data = torch.swapaxes(data, 1, 2)  # data of shape [batch_size, n_channels, input_length]
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=False)[1]
            # Filter out undefined
            pred = pred[target != -1]
            target = target[target != -1]
            count += target.shape[0]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            if TEST and batch_idx > 10:
                break
            if batch_idx % 1000 == 0:
                print('Test set:  Accuracy: {}/{} ({:.0f}%)'.format(
                    correct, count,
                    100. * correct / count))
        print("Final result: ")
        print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, count,
            100. * correct / count))

        confusion_matrix = confusion_matrix.numpy()
        precision, recall, f1 = get_confmat_metrics(confusion_matrix)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)
        # Save the result
        print("Saving result....")
        result_df = pd.DataFrame(data=[precision, recall, f1], columns=label_names, index=['precision','recall','f1'])
        result_df.to_csv(result_csv_path)
        print("Saving confusion mat...")
        confmat_df = pd.DataFrame(data=confusion_matrix, columns=label_names, index=label_names)
        confmat_df.to_csv(result_conf_path)

if __name__ == "__main__":
    make_predict()