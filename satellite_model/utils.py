import os
from re import S

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pandas as pd
from tqdm  import tqdm
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from torch.utils.data import Dataset

import torch
import random

import xarray as xr
import rioxarray

from train_utils import eval_metrics

def read_param_file(filepath):
    with open(filepath, "r") as f:
        output = f.read()
    return output

class PassthroughLoss:
    """use any normal torch loss function with
    the heteroscedastic network architecture.
    Simply ignores the second output."""
    def __init__(self, loss):
        self.loss = loss

    def __call__(self, y, y_hat):
         # assumes that y_hat has two components, corresponding to [mean, sigma2]
        if len(y_hat.shape) == 2:
            ym = y_hat[:, 0]
            prec = y_hat[:, 1]
        elif len(y_hat.shape) == 1:
            ym = y_hat[0]
            prec = y_hat[1]
        else:
            raise ValueError("wrong y_hat shape: " + str(y_hat.shape))

        return self.loss(y, ym)

def heteroscedastic_loss(y, y_hat):
    # assumes that y_hat has two components, corresponding to [mean, sigma2]
    if len(y_hat.shape) == 2:
        ym = y_hat[:, 0]
        prec = y_hat[:, 1]
    elif len(y_hat.shape) == 1:
        ym = y_hat[0]
        prec = y_hat[1]
    else:
        raise ValueError("wrong y_hat shape: " + str(y_hat.shape))

    ymd = (ym - y.squeeze())
    sigma2 = torch.exp(-prec)

    l = (0.5 * (sigma2 * ymd * ymd)) + 0.5 * prec #- 0.5 * torch.log(prec) #
    return l.sum()

def step(x, y, model, loss, optimizer, heteroscedastic):
    y_hat = model(x).squeeze()
    loss_epoch = loss(y.squeeze(), y_hat)
    if heteroscedastic:
        if len(y_hat.shape) == 2:
            y_hat = y_hat[:, 0]
        elif len(y_hat.shape) == 1:
            y_hat = y_hat[0]
        else:
            raise ValueError("wrong y_hat shape:" + str(y_hat.shape))


    optimizer.zero_grad()
    loss_epoch.backward()
    optimizer.step()

    # if len(dataset) % batch_size == 1 we might get singletons here
    # which is a problem in eval_metrics
    if len(y.shape) == 0:
        y = y.unsqueeze(0)
    if len(y_hat.shape) == 0:
        y_hat = y_hat.unsqueeze(0)

    #print("step y, y_hat shapes:", y.shape, y_hat.shape)
    metric_results = eval_metrics(y.detach().cpu(), y_hat.detach().cpu())

    return loss_epoch.detach().cpu(), metric_results

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_s5p_to_memory(sample, datadir, frequency, sources, s5p_dates):
    if sample.get("s5p") is None and "S5P" in sources:
        s5p_data = xr.open_dataset(os.path.join(datadir, "sentinel-5p", sample["s5p_path"])).rio.write_crs(4326)
        if frequency == "2018_2020":
            sample["s5p"] = s5p_data.tropospheric_NO2_column_number_density.values.squeeze()
        else:
            datestr = sample["date_str"]
            time_idx = np.where(s5p_dates==datestr)[0].item()
            sample["s5p"] = s5p_data.isel(time=time_idx).tropospheric_NO2_column_number_density.values.squeeze()

        s5p_data.close()
    return sample


def load_data(datadir, samples_file, frequency, sources):
    """load samples to memory, returns array of samples and array of stations
    each sample is a dict
    this version loads all samples from one station in one go (e.g. for multiple months), s.t. the S5P data for the station is only read once"""
    assert(sources in ["S2", "S2S5P"])
    assert(frequency in ["2018_2020", "monthly", "quarterly"])

    if not isinstance(samples_file, pd.DataFrame):
        samples_df = pd.read_csv(samples_file, index_col="idx")
    else:
        samples_df = samples_file
    samples_df = samples_df[np.isnan(samples_df.no2) == False]
    #samples_df = samples_df.iloc[0:100]
    #print(samples_df.shape)

    s5p_dates = None
    if frequency != "2018_2020":
        sample = samples_df.iloc[0]
        s5p_sample = xr.open_dataset(os.path.join(datadir, "sentinel-5p", sample["s5p_path"])).rio.write_crs(4326)
        if frequency == "quarterly":
            s5p_dates = np.array(["Q-" + str(dt.quarter) + "-" + str(dt.year) for dt in pd.to_datetime(s5p_sample.time.values)])
        elif frequency  == "monthly":
            s5p_dates = np.array([str(dt.month) + "-" + str(dt.year) for dt in pd.to_datetime(s5p_sample.time.values)])
        s5p_sample.close()

    samples = []
    stations = {}
    try:
        # here we assume that all S5P data for one station is stored in one .netcdf file
        # so it's faster to access the samples on a per station basis and only opening the
        # .netcdf file once
        for station in tqdm(samples_df.AirQualityStation.unique()):
            station_obs = samples_df[samples_df.AirQualityStation == station]
            s5p_path = station_obs.s5p_path.unique().item()
            s5p_data = xr.open_dataset(os.path.join(datadir, "sentinel-5p", s5p_path)).rio.write_crs(4326)

            for idx in station_obs.index.values:
                sample = samples_df.loc[idx].to_dict() # select by index value, not position
                sample["idx"] = idx
                if frequency == "2018_2020":
                    sample["s5p"] = s5p_data.tropospheric_NO2_column_number_density.values.squeeze()
                else:
                    datestr = sample["date_str"]
                    time_idx = np.where(s5p_dates==datestr)[0]
                    if len(time_idx) == 0:
                        print("No S5P data for", datestr)
                        continue
                    time_idx = time_idx.item()
                    sample["s5p"] = s5p_data.isel(time=time_idx).tropospheric_NO2_column_number_density.values.squeeze()

                samples.append(sample)
                stations[sample["AirQualityStation"]] = np.load(os.path.join(datadir, "sentinel-2", sample["img_path"]))

            s5p_data.close()

    except IndexError as e:
        print(e)
        print("idx:", idx)
        print()

    return samples, stations

def none_or_true(value):
    if value == 'None':
        return None
    elif value == "True":
        return True
    return value

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    