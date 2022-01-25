import os

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import sys
import copy
from tqdm  import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import NO2PredictionDataset
from transforms import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize
from model import get_model


def eval_metrics(y, y_hat):
    r2 = r2_score(y, y_hat)
    mae = mean_absolute_error(y, y_hat)
    mse = mean_squared_error(y, y_hat)

    return [r2, mae, mse]

def split_samples(samples, stations, test_size=0.2, val_size=0.2):
    stations_train, stations_test = train_test_split(stations, test_size=test_size)
    real_val_size = val_size / (1 - test_size)
    stations_train, stations_val = map(set, train_test_split(stations_train, test_size=real_val_size))
    stations_test = set(stations_test)

    samples_train = [s for s in samples if s["AirQualityStation"] in stations_train]
    samples_test = [s for s in samples if s["AirQualityStation"] in stations_test]
    samples_val = [s for s in samples if s["AirQualityStation"] in stations_val]

    # samples_train = copy.deepcopy([s for s in samples if s["AirQualityStation"] in stations_train])
    # samples_test = copy.deepcopy([s for s in samples if s["AirQualityStation"] in stations_test])
    # samples_val = copy.deepcopy([s for s in samples if s["AirQualityStation"] in stations_val])

    # samples_train = []
    # samples_test = []
    # samples_val = []

    # while samples:
        # sample = samples.pop()
        # station = sample["AirQualityStation"]
        # if station in stations_train:
            # samples_train.append(copy.deepcopy(sample))
        # elif station in stations_test:
            # samples_test.append(copy.deepcopy(sample))
        # elif station in stations_val:
            # samples_val.append(copy.deepcopy(sample))


    return samples_train, samples_val, samples_test, stations_train, stations_val, stations_test


def split_samples_df(samples, test_size=0.2, val_size=0.2):
    """split pd.DF s.t. all samples of a given station
    are either in the train or test set """
    stations = samples.AirQualityStation.unique()
    stations_train, stations_test = train_test_split(stations, test_size=test_size)
    real_val_size = val_size / (1 - test_size)
    stations_train, stations_val = map(set, train_test_split(stations_train, test_size=real_val_size))
    stations_test = set(stations_test)

    samples_train = samples[samples.AirQualityStation.isin(stations_train)]
    samples_val = samples[samples.AirQualityStation.isin(stations_val)]
    samples_test = samples[samples.AirQualityStation.isin(stations_test)]

    return samples_train, samples_val, samples_test

def train(sources, model, loss, optimizer, scheduler, dataloader, epochs, device, dropout, heteroscedastic):
    model.train()
    if dropout:
        model.head.turn_dropout_on()
    # train loop
    loss_history = []

    if "S5P" in sources:
        for epoch in range(epochs):
            loss_epoch = 0
            for idx, sample in enumerate(dataloader):
                img = sample["img"].float().to(device)
                s5p = sample["s5p"].float().unsqueeze(dim=1).to(device)
                y = sample["no2"].float().to(device)
                model_input = {"img" : img, "s5p" : s5p}
                y_hat = model(model_input).squeeze()
                loss_epoch = loss(y_hat, y)
                loss_epoch += loss_epoch.item()

                if heteroscedastic:
                    if len(y_hat.shape) == 2:
                        y_hat = y_hat[:, 0]
                    if len(y_hat.shape) == 1:
                        y_hat = y_hat[0]
                    else:
                        raise ValueError()

                optimizer.zero_grad()
                loss_epoch.backward()
                optimizer.step()
            scheduler.step(epoch)
            torch.cuda.empty_cache()
            loss_history.append(loss_epoch/idx) # save the epochs average loss
#            print("Epoch", epoch, "loss:", loss_epoch.cpu().detach().numpy().item()/idx)
    else:
        for epoch in range(epochs):
            loss_epoch = 0
            for idx, sample in enumerate(dataloader):
                img = sample["img"].float().to(device)
                y = sample["no2"].float().to(device)
                y_hat = model(img).squeeze()
                loss_epoch = loss(y_hat, y)
                loss_epoch += loss_epoch.item()
                optimizer.zero_grad()
                loss_epoch.backward()
                optimizer.step()
            scheduler.step(epoch)
            torch.cuda.empty_cache()
            loss_history.append(loss_epoch/idx) # save the epochs average loss
#            print("Epoch", epoch, "loss:", loss_epoch.cpu().detach().numpy().item()/idx)

    return model

def test(sources, model, dataloader, device, datastats, dropout, heteroscedastic):
    # test
    model.eval()
    if dropout:
        model.head.turn_dropout_on(use=False)
    predictions = []
    measurements = []

    with torch.no_grad():
        if "S5P" in sources:
            for idx, sample in enumerate(dataloader):
                img = sample["img"].float().to(device)
                s5p = sample["s5p"].float().unsqueeze(dim=1).to(device)
                y = sample["no2"].float().to(device).squeeze()
                model_input = {"img" : img, "s5p" : s5p}
                y_hat = model(model_input).squeeze()

                if heteroscedastic:
                    if len(y_hat.shape) == 2:
                        y_hat = y_hat[:, 0]
                    elif len(y_hat.shape) == 1:
                        y_hat = y_hat[0]
                    else:
                        raise ValueError()

                measurements.append(y.cpu().numpy().item())
                predictions.append(y_hat.cpu().numpy().item())
        else:
            for idx, sample in enumerate(dataloader):
                img = sample["img"].float().to(device)
                y = sample["no2"].float().to(device).squeeze()
                y_hat = model(img).squeeze()

                if heteroscedastic:
                    if len(y_hat.shape) == 2:
                        y_hat = y_hat[:, 0]
                    elif len(y_hat.shape) == 1:
                        y_hat = y_hat[0]
                    else:
                        raise ValueError()

                measurements.append(y.cpu().numpy().item())
                predictions.append(y_hat.cpu().numpy().item())

    measurements = np.array(measurements)
    predictions = np.array(predictions)

    # predictions = Normalize.undo_no2_standardization(datastats, predictions)
    # measurements = Normalize.undo_no2_standardization(datastats, measurements)

    return measurements, predictions

if __name__ == "__main__":
    from utils import load_data
    # call as `python train.py data/samples_S2_whole_timespan.csv /netscratch/lscheibenreif/eea`
    samples_file = sys.argv[1]
    datadir = sys.argv[2]
    verbose = sys.argv[3]
    sources = samples_file.split("_")[1]
    if "2018_2020" in samples_file:
        frequency = "2018_2020"
    else:
        frequency = samples_file.split("_")[2].replace(".csv", "")

    if verbose: print("Loading data...")
    samples, stations = load_data(datadir, samples_file, frequency, sources)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #if verbose:
    #    print("device:", device)
    #    print("frequency:", frequency)
    #    print("sources:", sources)

    # pre-trained checkpoint for landuse on bigearthnet
    checkpoint = "checkpoints/pretrained_resnet50_LUC.model"
    # checkpoint = None

    loss = nn.MSELoss()
    epochs = 20
    batch_size = 50
    datastats = DatasetStatistics()
    tf = transforms.Compose([ChangeBandOrder(), Normalize(datastats), Randomize(), ToTensor()])

    num_runs = 20
    performances_test = []
    performances_val = []
    performances_train = []
    for run in tqdm(range(num_runs)):

        samples_train, samples_val, samples_test = split_samples(samples, stations, 0.2, 0.2)
        if verbose:
            print("run:", run)
            print("train size:", len(samples_train))
            print("test size:", len(samples_test))
            print("val size:", len(samples_val))

        dataset_test = NO2PredictionDataset(datadir, samples_test, frequency, sources, transforms=tf)
        dataset_train = NO2PredictionDataset(datadir, samples_train, frequency, sources, transforms=tf)
        dataset_val = NO2PredictionDataset(datadir, samples_val, frequency, sources, transforms=tf)

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
        dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)
        dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)
        dataloader_train_for_testing = DataLoader(dataset_train, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)

        model = get_model(sources, device, checkpoint)
        model.to(device)

        # optimize only the head
        # optimizer = optim.SGD(model.head.parameters(), lr=0.004)

        # optimize entire model (head + backbone)
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, threshold=1e-5, min_lr=1e-6)

        for epoch in range(epochs):

            model = train(sources, model, loss, optimizer, scheduler, dataloader_train, 1, device)
            val_y, val_y_hat = test(sources, model, dataloader_val, device, datastats)

            valid_val = (val_y < 100) & (val_y > 0)

            eval_val = eval_metrics(val_y, val_y_hat)

            if epoch > 5 and sum(valid_val) > len(valid_val) - 5:
                if eval_val[1] > np.mean([performances_val[-3][3], performances_val[-2][3], performances_val[-1][3]]):
                    # performance on evaluation set is decreasing
                    if verbose: print("Stop at epoch:", epoch)
                    break

            performances_val.append([run, epoch] + eval_val)

        test_y, test_y_hat = test(sources, model, dataloader_test, device, datastats)
        train_y, train_y_hat = test(sources, model, dataloader_train_for_testing, device, datastats)

        valid = (test_y < 100) & (test_y > 0)
        valid_train = (train_y < 100) & (train_y > 0)

        eval_test = eval_metrics(test_y, test_y_hat)
        eval_train = eval_metrics(train_y, train_y_hat)

        performances_test.append(eval_test)
        performances_train.append(eval_train)

        if verbose:
            print("\tvalid_train sum,len:", sum(valid_train), len(valid_train))
            print("\tvalid sum,len:", sum(valid), len(valid))
            print("\ttrain scores:", *eval_train)
            print("\ttest scores:", *eval_test)
            #print("\tval scores:", *eval_val)

    performances_train = pd.DataFrame(performances_train, columns=["r2", "mae", "mse"])
    performances_test = pd.DataFrame(performances_test, columns=["r2", "mae", "mse"])
    performances_val = pd.DataFrame(performances_val, columns=["run", "epoch", "r2", "mae", "mse"])

    performances_train.to_csv("data/performances_train_" + sources + "_" + frequency + ".csv", index=False)
    performances_test.to_csv("data/performances_test_" + sources + "_" + frequency + ".csv", index=False)
    performances_val.to_csv("data/performances_val_" + sources + "_" + frequency + ".csv", index=False)

