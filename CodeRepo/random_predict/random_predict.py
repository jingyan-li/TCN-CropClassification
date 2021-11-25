from tqdm import tqdm
import wandb
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import argparse
import sys
sys.path.append("../../")
import os
from simple_tcn.model import TCN
from utils.dataset import Dataset
from simple_tcn.config import config
import sklearn.metrics as metrics
import pickle

def calculate_scores(cof_mat):
    precision = np.diagonal(cof_mat) / np.sum(cof_mat, axis=0)  # TP/P
    recall = np.diagonal(cof_mat) / np.sum(cof_mat, axis=1)  # TP/T
    f1 = 2 * precision * recall / (precision + recall)
    overall_accuracy = np.sum(np.diagonal(cof_mat)) / np.sum(cof_mat)
    return precision, recall, f1, overall_accuracy

if __name__ == "__main__":
    MODEL_TITLE = "random_prediction"
    data_path = config["data-path"]
    batch_size = config["batch_size"]
    label_names = config["label-names"]
    LOG_PATH = "log/"
    SAMPLE_VAL = False
    SAMPLESIZE = 0.01  # x% data will be used

    input_channels = 4

    # Calculate the weights of each label
    dataset = Dataset(path=data_path, time_downsample_factor=1, num_channel=input_channels)
    gt_list = dataset.return_labels()
    labels, pix_counts = np.unique(gt_list, return_counts=True)
    # inds = pix_counts.argsort()
    # pix_counts_sorted = pix_counts[inds]
    # labels_sorted = labels[inds]
    # label_names_sorted = [label_names[labels.tolist().index(x)] for x in labels_sorted]

    weights = pix_counts / np.sum(pix_counts)

    RANDOM_SPLIT = int(dataset.__len__() * 0.8)
    train_dset, val_dset = random_split(dataset, [RANDOM_SPLIT, dataset.__len__() - RANDOM_SPLIT])
    print(f"It contains {len(val_dset)} windows.")
    if SAMPLE_VAL:
        sampling = np.random.randint(len(val_dset), size=round(len(val_dset) * SAMPLESIZE))
        val_dset = torch.utils.data.Subset(val_dset, sampling)
    print(f"VALIDATION dataset contains {len(val_dset)} windows")

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size)


    cof_mat = np.zeros((49, 49))
    for x, y in tqdm(val_loader):
        x = torch.transpose(x, 1, 2)
        x, y = Variable(x), Variable(y)
        y_pred = np.random.choice(a=len(labels), size=y.shape, p=weights)
        cof_mat += metrics.confusion_matrix(y, y_pred, labels=labels)

    precision, recall, f1, overall_accuracy = calculate_scores(cof_mat)

    print(f"Final validation score: \nprecision - {precision}\nrecall - {recall}\nf1 - {f1}")
    print(f"Overall accuracy: {overall_accuracy}")
    # Save scores
    if not os.path.exists(os.path.join(LOG_PATH, MODEL_TITLE)):
        os.makedirs(os.path.join(LOG_PATH, MODEL_TITLE))
    np.save(os.path.join(LOG_PATH, MODEL_TITLE, "precision"), precision)
    np.save(os.path.join(LOG_PATH, MODEL_TITLE, "recall"), recall)
    np.save(os.path.join(LOG_PATH, MODEL_TITLE, "f1"), f1)
    np.save(os.path.join(LOG_PATH, MODEL_TITLE, "cof_mat"), cof_mat)