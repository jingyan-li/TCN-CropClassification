from simple_tcn.model import TCN
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from simple_tcn.config import config
import sys
sys.path.append("../../")
import optuna
from utils.dataset import Dataset
from torch.utils.data import random_split
import pickle
import torch.optim as optim
import numpy as np
from datetime import datetime
import logging


CUDA = True
batch_size = 64
log_interval = 500
data_path = config["data-path"]
# dropout = 0.05
# label_path = config["label-path"]
optimizer = config["optim"]
epochs = 2
SEED = 2021

# Fix seed for reproducing
torch.manual_seed(SEED)
# Use cuda
if torch.cuda.is_available():
    print("CUDA is available")
    if not CUDA:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


n_classes = 13
input_channels = 4
seq_length = 71
steps = 0
# label_weights = []
# with open(label_path, "rb") as f:
#     label_weights = pickle.load(f)
# label_weights = torch.Tensor(label_weights)
label_names = config["label-names"]


def train(ep, model, optimizer, train_loader):
    global steps
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data = torch.transpose(data, 1, 2)  # data of shape [batch_size, n_channels, input_length]
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        # compute output and loss
        output = model(data)
        loss = F.cross_entropy(output, target)
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update loss
        train_loss += loss
        steps += 1
        if batch_idx > 0 and batch_idx % log_interval == 0:
            if batch_idx >= int(len(train_loader.dataset)/batch_size):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                    ep, len(train_loader.dataset), len(train_loader.dataset),
                        100. * len(train_loader) / len(train_loader), train_loss.item()/log_interval, steps))
                break
            else:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                    ep, (batch_idx+1) * batch_size, len(train_loader.dataset),
                    100. * (batch_idx+1) / len(train_loader), train_loss.item()/log_interval, steps))

            # wandb.log({"loss": train_loss.item()/log_interval})
            train_loss = 0
        # torch.save(model.state_dict(), 'params.pkl')


def validation(model, val_loader):
    # load model
    # model.load_state_dict(torch.load('params.pkl'))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(val_loader)):
            if CUDA:
                data, target = data.cuda(), target.cuda()
            data = torch.transpose(data, 1, 2)  # data of shape [batch_size, n_channels, input_length]
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.data.max(1, keepdim=False)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            # if batch_idx >= int(len(val_loader.dataset) / batch_size):
            #     print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
            #         len(val_loader.dataset), len(val_loader.dataset),
            #         100. * len(val_loader) / len(val_loader), test_loss/(batch_idx+1)))
            #     break
            # else:
            #     print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
            #         (batch_idx + 1) * batch_size, len(val_loader.dataset),
            #             100. * (batch_idx + 1) / len(val_loader), test_loss/(batch_idx+1)))

        test_loss /= batch_idx + 1
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        return 100. * correct / len(val_loader.dataset)


def start_train_and_test(trial, epochs, model, optimizer, train_loader, val_loader):
    test_acc: list = []
    for ep in range(1, epochs + 1):
        train(ep, model, optimizer, train_loader)
        tacc = validation(model, val_loader)
        test_acc.append(tacc)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(np.array(test_acc))


def objective(trail: optuna.Trial):
    print("Produce data...")
    dataset = Dataset(path=data_path, time_downsample_factor=1, num_channel=input_channels)
    RANDOM_SPLIT = int(dataset.__len__() * 0.8)
    train_dset, val_dset = random_split(dataset, [RANDOM_SPLIT, dataset.__len__() - RANDOM_SPLIT])
    print(f"Train_dset contains {len(train_dset)} pixels.")
    print(f"Val_dset contains {len(val_dset)} pixels.")

    # Sampling subset of train and validation set to speed up the hyperparameter tuning
    SAMPLESIZE = 0.3
    sampling_train = np.random.randint(len(train_dset), size=round(len(train_dset) * SAMPLESIZE))
    train_dset = torch.utils.data.Subset(train_dset, sampling_train)
    sampling_val = np.random.randint(len(val_dset), size=round(len(val_dset) * SAMPLESIZE))
    val_dset = torch.utils.data.Subset(train_dset, sampling_val)
    print(f"After sampling, train_dset contains {len(train_dset)} pixels.")
    print(f"After sampling, val_dset contains {len(val_dset)} pixels.")

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, num_workers=0)

    n_levels = trail.suggest_int('levels', 5, 8)
    n_hunits = trail.suggest_int('nhid', 25, 50)
    output_channels = [n_hunits]*n_levels
    lr = trail.suggest_loguniform('lr', 1e-5, 1e-1)
    kernel_size = trail.suggest_int('kernel_size', 3, 8)
    dropout = trail.suggest_int('dropout', 0.03, 0.2)


    model = TCN(input_channels, n_classes, output_channels, kernel_size=kernel_size, dropout=dropout)
    if CUDA:
        model.cuda()

    optimizer = getattr(optim, config["optim"])(model.parameters(), lr=lr)

    acc = start_train_and_test(trail, epochs, model, optimizer, train_loader, val_loader)
    return acc


def main():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=15)

    # with open(f"Hyperpara_{dt.month}.{dt.day}_{dt.hour}.{dt.minute}", "w") as f:
    #     f.write(f"Number of finished trials: {len(study.trials)} \n")

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    dt = datetime.now()
    fig.savefig(f"Fig_{dt.month}.{dt.day}_{dt.hour}.{dt.minute}.png")


if __name__ == '__main__':
    main()