import os

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import sys
import copy
import random
import argparse
from datetime import datetime
from distutils.util import strtobool

import mlflow
import numpy as np
import pandas as pd
from tqdm  import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import NO2PredictionDataset
from transforms import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize
from model import get_model
from utils import load_data, none_or_true, dotdict, set_seed, step, heteroscedastic_loss, PassthroughLoss
from train_utils import eval_metrics, split_samples, train, test

bool_args = ["verbose",
            "early_stopping",
            "heteroscedastic",
            ]

parser = argparse.ArgumentParser(description='train_s2s5p_model')

# parameters
parser.add_argument('--samples_file', default="../data/samples_S2S5P_2018_2020_eea.csv", type=str)
parser.add_argument('--datadir', default="/netscratch/lscheibenreif/eea", type=str)
parser.add_argument('--verbose', default="True", type=str)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--runs', default=1, type=int)
parser.add_argument('--result_dir', default="results/", type=str)
parser.add_argument('--checkpoint', default="../checkpoints/pretrained_resnet50_LUC.model", type=str)
parser.add_argument('--early_stopping', default="False", type=str)
parser.add_argument('--weight_decay_lambda', default=0.001, type=float)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--dropout', default=None, type=none_or_true)
parser.add_argument('--dropout_p_second_to_last_layer', default=0.0, type=float)
parser.add_argument('--dropout_p_last_layer', default=0.0, type=float)
parser.add_argument('--heteroscedastic', default="False", type=str)

args = parser.parse_args()
config = dotdict({k : strtobool(v) if k in bool_args else v for k,v in vars(args).items()})

sources = config.samples_file.split("_")[1]
frequency = "2018_2020" if "2018" in config.samples_file else config.samples_file.split("_")[2].replace(".csv", "")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_name = "pretrained" if config.checkpoint is not None else "from_scratch"

experiment = "_".join([datetime.today().strftime('%Y-%m-%d-%H:%M'), sources, checkpoint_name, frequency])
if config.verbose: print("Initializing mlflow experiment:", experiment)
experiment_id = mlflow.create_experiment(experiment)

if config.verbose:
    print(config.samples_file)
    print(config.datadir)
    print(sources)
    print(frequency)
    print(config.checkpoint)
    print(device)
    print("Start loading samples...")

samples, stations = load_data(config.datadir, config.samples_file, frequency, sources)

if config.heteroscedastic:
    msel = nn.MSELoss()
    loss = PassthroughLoss(msel)
    print("Start heteroscedastic model training with MSELoss")
else:
    loss = nn.MSELoss()

datastats = DatasetStatistics()
tf = transforms.Compose([ChangeBandOrder(), Normalize(datastats), Randomize(), ToTensor()])

performances_test = []
performances_val = []
performances_train = []

for run in tqdm(range(1, config.runs+1), unit="run"):

    # fix a different seed for each run
    seed = run

    with mlflow.start_run(experiment_id=experiment_id):
        os.system("rm artifacts/*") # delete last run's artifacts
        mlflow.log_param("samples_file", config.samples_file)
        mlflow.log_param("heteroscedastic", config.heteroscedastic)
        mlflow.log_param("datadir", config.datadir)
        mlflow.log_param("sources", sources)
        mlflow.log_param("frequency", frequency)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("result_dir", config.result_dir)
        mlflow.log_param("pretrained_checkpoint", config.checkpoint)
        mlflow.log_param("device", device)
        mlflow.log_param("early_stopping", config.early_stopping)
        mlflow.log_param("learning_rate", config.learning_rate)
        mlflow.log_param("run", run)
        mlflow.log_param("dropout", config.dropout)
        mlflow.log_param("dropout_p_second_to_last_layer", config.dropout_p_second_to_last_layer)
        mlflow.log_param("dropout_p_last_layer", config.dropout_p_last_layer)
        mlflow.log_param("weight_decay", config.weight_decay_lambda)
        mlflow.log_param("epochs", config.epochs)
        mlflow.log_param("seed", seed)

        # set the seed for this run
        set_seed(seed)

        if config.dropout:
            dropout_config = {
                    "p_second_to_last_layer" : config.dropout_p_second_to_last_layer,
                    "p_last_layer" : config.dropout_p_last_layer,
                    }
        else:
            dropout_config = None

        # initialize dataloaders + model
        if config.verbose: print("Initializing dataset")
        samples_train, samples_val, samples_test, stations_train, stations_val, stations_test = split_samples(samples, list(stations.keys()), 0.2, 0.2)
        if config.verbose: print("First stations_train:", list(stations_train)[:10])
        dataset_test = NO2PredictionDataset(config.datadir, samples_test, frequency, sources, transforms=tf, station_imgs=stations)
        dataset_train = NO2PredictionDataset(config.datadir, samples_train, frequency, sources, transforms=tf, station_imgs=stations)
        dataset_val = NO2PredictionDataset(config.datadir, samples_val, frequency, sources, transforms=tf, station_imgs=stations)
        dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, num_workers=4, shuffle=True, pin_memory=False)
        dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=1, shuffle=False, pin_memory=False)
        dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=1, shuffle=False, pin_memory=False)
        dataloader_train_for_testing = DataLoader(dataset_train, batch_size=1, num_workers=1, shuffle=False, pin_memory=False)

        if config.verbose: print("Initializing model")
        model = get_model(sources, device, config.checkpoint, dropout=dropout_config, heteroscedastic=config.heteroscedastic)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay_lambda)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5, threshold=1e6, min_lr=1e-7, verbose=True)

        if config.verbose: print("Start training")
        # train the model
        for epoch in range(config.epochs):
            model.train()
            if config.dropout:
                model.head.turn_dropout_on()

            loss_history = []
            loss_epoch = []
            r2_epoch = []
            mae_epoch = []
            mse_epoch = []

            if epoch == 5 and config.heteroscedastic:
                if config.verbose: print("Changing to heteroscedastic loss...")
                loss = heteroscedastic_loss
            for idx, sample in enumerate(dataloader_train):
                model_input = sample["img"].float().to(device)
                if "S5P" in sources:
                    s5p = sample["s5p"].float().unsqueeze(dim=1).to(device)
                    model_input = {"img" : model_input, "s5p" : s5p}
                y = sample["no2"].float().to(device)

                loss_batch, metric_results = step(model_input, y, model, loss, optimizer, config.heteroscedastic)
                loss_epoch.append(loss_batch.item())
                r2_epoch.append(metric_results[0])
                mae_epoch.append(metric_results[1])
                mse_epoch.append(metric_results[2])

            loss_epoch = np.array(loss_epoch).mean()
            r2_train_epoch = np.array(r2_epoch).mean()
            mae_train_epoch = np.array(mae_epoch).mean()
            mse_train_epoch = np.array(mse_epoch).mean()

            scheduler.step(loss_epoch)
            torch.cuda.empty_cache()
            loss_history.append(loss_epoch)

            val_y, val_y_hat = test(sources, model, dataloader_val, device, datastats, config.dropout, config.heteroscedastic)

            valid_val = (val_y_hat < 100) & (val_y_hat > 0)
            eval_val = eval_metrics(val_y, val_y_hat)

            if config.verbose: print("Fraction of valid estimates:", sum(valid_val)/len(valid_val))

            if config.early_stopping:
                # stop training if evaluation performance does not increase
                if epoch > 25 and sum(valid_val) > len(valid_val) - 5:
                    if eval_val[0] > np.mean([performances_val[-3][2], performances_val[-2][2], performances_val[-1][2]]):
                        # performance on evaluation set is decreasing
                        if config.verbose: print(f"Early stop at epoch: {epoch}")
                        mlflow.log_param("early_stop_epoch", epoch)
                        break

            if config.verbose: print(f"Epoch: {epoch}, {eval_val}")
            performances_val.append([run, epoch] + eval_val)
            mlflow.log_metrics({"val_r2_epoch" : eval_val[0], "val_mae_epoch" : eval_val[1], "val_mse_epoch" : eval_val[2]}, step=epoch)
            mlflow.log_metrics({"train_loss_epoch" : loss_epoch, "train_r2_epoch" : r2_train_epoch, "train_mae_epoch" : mae_train_epoch, "train_mse_epoch" : mse_train_epoch}, step=epoch)
            mlflow.log_metric("current_epoch", epoch, step=epoch)


        torch.save(model.state_dict(), "artifacts/model_state.model")
        os.system("cp model.py artifacts/model.py") # save the model definition along with the trained model
        for name, station_list in [("stations_train", stations_train), ("stations_val", stations_val), ("stations_test", stations_test)]:
            # save the dataset train/val/test split
            with open("artifacts/" + name + ".txt", "w") as f:
                for station in station_list:
                    f.write("%s\n" % station)

        test_y, test_y_hat = test(sources, model, dataloader_test, device, datastats, config.dropout, config.heteroscedastic)
        train_y, train_y_hat = test(sources, model, dataloader_train_for_testing, device, datastats, config.dropout, config.heteroscedastic)

        valid = (test_y_hat < 100) & (test_y_hat > 0)
        valid_train = (train_y_hat < 100) & (train_y_hat > 0)

        eval_test = eval_metrics(test_y, test_y_hat)
        eval_train = eval_metrics(train_y, train_y_hat)

        # save img of predictions as artifact
        img, (ax1,ax2) = plt.subplots(1,2, figsize=(12,7))
        for ax in (ax1,ax2):
            ax.set_xlim((0,100))
            ax.set_ylim((0,100))
            ax.plot((0,0),(100,100), c="red")
        ax1.scatter(test_y, test_y_hat, s=2)
        ax1.set_xlabel("Measurements")
        ax1.set_ylabel("Predictions")
        ax1.set_title("test")
        ax2.scatter(train_y, train_y_hat, s=2)
        ax2.set_title("train")
        ax2.set_xlabel("Measurements")
        ax2.set_ylabel("Predictions")
        plt.savefig("artifacts/predictions.png")
        #mlflow.log_figure(img, "imgs/predictions.png")

        mlflow.log_metric("test_r2", eval_test[0])
        mlflow.log_metric("test_mae", eval_test[1])
        mlflow.log_metric("test_mse", eval_test[2])

        performances_test.append(eval_test)
        performances_train.append(eval_train)

        mlflow.log_artifacts("artifacts") # log everything that was written to the artifacts directory

    performances_val = pd.DataFrame(performances_val, columns=["run", "epoch", "r2", "mae", "mse"])
    performances_test = pd.DataFrame(performances_test, columns=["r2", "mae", "mse"])
    performances_train = pd.DataFrame(performances_train, columns=["r2", "mae", "mse"])


if config.checkpoint is not None: checkpoint_name = config.checkpoint.split("/")[1].split(".")[0]

# save results
if config.verbose: print("Writing results...")
performances_test.to_csv(os.path.join(config.result_dir, "_".join([sources, str(checkpoint_name), frequency, "test", str(config.epochs), "epochs"]) + ".csv"), index=False)
performances_train.to_csv(os.path.join(config.result_dir, "_".join([sources, str(checkpoint_name), frequency, "train", str(config.epochs), "epochs"]) + ".csv"), index=False)
performances_val.to_csv(os.path.join(config.result_dir, "_".join([sources, str(checkpoint_name), frequency, "val", str(config.epochs), "epochs"]) + ".csv"), index=False)

# save the model
if config.verbose: print("Writing model...")
torch.save(model.state_dict(), os.path.join(config.result_dir, "_".join([sources, str(checkpoint_name), frequency, str(config.epochs), "epochs"]) + ".model"))
if config.verbose: print("done.")
