# Validate test dataset
import tqdm
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
from simple_tcn.model import TCN
from simple_tcn.config import config
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
label_path = config["label-path"]
result_csv_path = os.path.join(checkpoint_path, checkpoint_name[:-3]+"_test_metrics.csv")
# Load label weights
label_weights = None
useLabelWeights = config["useLabelWeight"]
if useLabelWeights:
    with open(label_path, "rb") as f:
        label_counts = pickle.load(f)
    label_weights = torch.Tensor(
        get_label_weights(config["label-weight-method"],
                          label_counts,
                          config["label-weight-beta"])
    )
label_names = config["label-names"]


n_classes = 13
input_channels = 4
output_channels = [n_hunits]*n_levels
# Model
model = TCN(input_channels, n_classes, output_channels, kernel_size=kernel_size, dropout=dropout)

# Load checkpoint
checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_name))
model.load_state_dict(checkpoint['model_state_dict'])


def make_predict(val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        confusion_matrix = torch.zeros(n_classes, n_classes)
        for batch_idx, (data, target) in tqdm(enumerate(val_loader)):
            if CUDA:
                data, target = data.cuda(), target.cuda()
                if useLabelWeights:
                    label_weights = label_weights.cuda()
            data = torch.swapaxes(data, 1, 2)  # data of shape [batch_size, n_channels, input_length]
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, weight=label_weights).item()
            pred = output.data.max(1, keepdim=False)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            if TEST and batch_idx > 10:
                break

        test_loss /= batch_idx+1
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        confusion_matrix = confusion_matrix.numpy()
        precision, recall, f1 = get_confmat_metrics(confusion_matrix)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)
        # Save the result
        print("Saving result....")
        result_df = pd.DataFrame(data=[precision, recall, f1], columns=label_names, index=['precision','recall','f1'])
        result_df.to_csv(result_csv_path)

if __name__ == "__main__":
    test_dset = Dataset(path="D:\jingyli\II_Lab3\data\imgint_testset_v2.hdf5",
                        time_downsample_factor=1,
                        num_channel=input_channels)

    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, num_workers=0)

    make_predict(test_loader)