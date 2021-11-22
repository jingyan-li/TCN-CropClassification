import pickle

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

# Visualize training
wandb.init(project="ii-lab3", entity="jejejennea")


# Read configs
SEED = config["seed"]
CUDA = config["cuda"]
batch_size = config["batch_size"]
epochs = config["epochs"]
kernel_size = config["kernel-size"]
dropout = config["dropout"]
lr = config["learning-rate"]
optimizer = config["optim"]

log_interval = config["log-interval"]

n_hunits = config["nhid"]
n_levels = config["levels"]

data_path = config["data-path"]
cp_path = config["checkpoint-path"]
label_path = config["label-path"]
if not os.path.exists(cp_path):
    os.makedirs(cp_path)

# Add config to wandb
wandb.config = {
  "learning_rate": lr,
  "epochs": epochs,
  "batch_size": batch_size,
    "kernel_size": kernel_size,
    "levels": n_levels,
    "hunits": n_hunits
}
# Fix seed for reproducing
torch.manual_seed(SEED)
# Use cuda
if torch.cuda.is_available():
    print("CUDA is available")
    if not CUDA:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# Initiate variables
n_classes = 52  # TODO: No.classes = 49? or use 52?
input_channels = 4
output_channels = [n_hunits]*n_levels   # TODO: Hidden units (channels)
seq_length = 71  # Temporal length per sample
steps = 0

with open(label_path, "rb") as f:
    label_weights = pickle.load(label_path)
label_weights = torch.Tensor(label_weights)
label_names = config["label-names"]
print(config)

# Data loader
dataset = Dataset(path=data_path, time_downsample_factor=1, num_channel=input_channels)
RANDOM_SPLIT = int(dataset.__len__()*0.8)
train_dset, val_dset = random_split(dataset, [RANDOM_SPLIT, dataset.__len__()-RANDOM_SPLIT])
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size)
# Model
model = TCN(input_channels, n_classes, output_channels, kernel_size=kernel_size, dropout=dropout)

if CUDA:
    model.cuda()

optimizer = getattr(optim, optimizer)(model.parameters(), lr=lr)


def train(ep):
    global steps
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data = torch.swapaxes(data, 1, 2)  # data of shape [batch_size, n_channels, input_length]
        data, target = Variable(data), Variable(target)
        # compute output and loss
        output = model(data)
        # TODO: Change loss function, add weights
        loss = F.nll_loss(output, target, weight=label_weights)
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update loss
        train_loss += loss
        # TODO: Change steps
        steps += seq_length
        if batch_idx > 0 and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/log_interval, steps))
            wandb.log({"loss": train_loss.item()/log_interval})
            train_loss = 0


def validation():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(val_loader):
            if CUDA:
                data, target = data.cuda(), target.cuda()
            data = torch.swapaxes(data, 1, 2)  # data of shape [batch_size, n_channels, input_length]
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False, weight=label_weights).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # Log confusion matrix
            wandb.log({"val_conf_mat":
                        wandb.plot.confusion_matrix(probs=None,
                                                    y_true=target.data.cpu(),
                                                    preds=pred.data.cpu(),
                                                    class_names=label_names)})
        test_loss /= len(val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        wandb.log({"val_loss": test_loss})
        # TODO: Log f1 score, confusion mat, overall accuracy
        return test_loss


if __name__ == "__main__":
    min_loss = 100
    for epoch in range(1, epochs+1):
        train(epoch)
        val_loss = validation()
        # TODO: Add early stop
        if val_loss < min_loss:
            min_loss = val_loss
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': min_loss,
            }, os.path.join(cp_path, f"best-epoch{epoch}.pt"))
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # Save final checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': min_loss,
    }, os.path.join(cp_path, f"last.pt"))