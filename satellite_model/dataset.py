import os

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

import xarray as xr
import rioxarray

class NO2PredictionDataset(Dataset):

    def __init__(self, datadir, samples, frequency, sources, transforms=None, station_imgs=None):
        assert(sources in ["S2", "S2S5P"])
        assert(frequency in ["2018_2020", "monthly", "quarterly"])

        self.datadir = datadir
        self.transforms = transforms
        self.frequency = frequency
        self.sources = sources
        self.station_imgs = station_imgs # dict of AirQualityStation -> S2 image

        self.samples = samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.station_imgs is not None:
            sample["img"] = self.station_imgs.get(sample["AirQualityStation"])
            
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.samples)

    def display_sample(self, sample, title=None):
        img = sample["img"]
        band_data = self._normalize_for_display(img)

        if "S5P" in self.sources:
            fig, axs = plt.subplots(1, 2, figsize=(7,7))
            s2_ax = axs[0]
        else:
            fig, s2_ax = plt.subplots(1, figsize=(5,5))
        s2_ax.imshow(band_data[:, :, [3,2,1]])
        s2_ax.set_title("Sentinel2 data")

        if "S5P" in self.sources:
            im = axs[1].imshow(sample["s5p"])
            axs[1].set_title("Sentinel-5P data")
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

        if title is not None:
            fig.suptitle(title)

        plt.show()

    def _normalize_for_display(self, band_data):
        band_data = reshape_as_image(np.array(band_data))
        lower_perc = np.percentile(band_data, 2, axis=(0,1))
        upper_perc = np.percentile(band_data, 98, axis=(0,1))
        return (band_data - lower_perc) / (upper_perc - lower_perc)


class NO2PredictionDatasetWithDataloading(Dataset):

    def __init__(self, datadir, samples, frequency, sources, preload=False, transforms=None):
        assert(sources in ["S2", "S2S5P"])
        assert(frequency in ["2018_2020", "monthly", "quarterly"])

        self.datadir = datadir
        self.transforms = transforms
        self.frequency = frequency
        self.sources = sources

        self.sample_df = samples
        self.preload = preload

        # set the dates only once
        if self.frequency != "2018_2020":
            sample = self.sample_df.iloc[0]
            s5p_sample = xr.open_dataset(os.path.join(self.datadir, "sentinel-5p", sample["s5p_path"])).rio.write_crs(4326)
            if self.frequency == "quarterly":
                self.s5p_dates = np.array(["Q-" + str(dt.quarter) + "-" + str(dt.year) for dt in pd.to_datetime(s5p_sample.time.values)])
            elif self.frequency  == "monthly":
                self.s5p_dates = np.array([str(dt.month) + "-" + str(dt.year) for dt in pd.to_datetime(s5p_sample.time.values)])
            s5p_sample.close()

        if preload:
            self.sample_list = self.preload_samples()

    def __getitem__(self, idx):
        if self.preload:
            sample = self.sample_list[idx]
        else:
            sample = self.sample_df.iloc[idx].to_dict()
            sample["idx"] = idx
            sample = self.load_data_to_memory(sample)

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return self.sample_df.shape[0]

    def preload_samples(self):
        samples = []
        print("Preloading data to memory")
        for idx in tqdm(range(self.sample_df.shape[0])):
            sample = self.sample_df.iloc[idx].to_dict()
            sample["idx"] = idx
            sample = self.load_data_to_memory(sample)
            samples.append(sample)

        return samples

    def load_data_to_memory(self, sample):
        if sample.get("img") is None:
            sample["img"] = np.load(os.path.join(self.datadir, "sentinel-2", sample["img_path"]))

        if sample.get("s5p") is None and "S5P" in self.sources:
            s5p_data = xr.open_dataset(os.path.join(self.datadir, "sentinel-5p", sample["s5p_path"])).rio.write_crs(4326)
            if self.frequency == "2018_2020":
                sample["s5p"] = s5p_data.tropospheric_NO2_column_number_density.values.squeeze()
            else:
                datestr = sample["date_str"]
                time_idx = np.where(self.s5p_dates==datestr)[0].item()
                sample["s5p"] = s5p_data.isel(time=time_idx).tropospheric_NO2_column_number_density.values.squeeze()

            s5p_data.close()
        return sample

    def display_sample(self, sample, title=None):
        img = sample["img"]
        band_data = self._normalize_for_display(img)

        if "S5P" in self.sources:
            fig, axs = plt.subplots(1, 2, figsize=(7,7))
        else:
            fig, axs = plt.subplots(1, figsize=(5,5))
        axs[0].imshow(band_data[:, :, [3,2,1]])
        axs[0].set_title("Sentinel2 data")

        if "S5P" in self.sources:
            im = axs[1].imshow(sample["s5p"])
            axs[1].set_title("Sentinel-5P data")
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

        if title is not None:
            fig.suptitle(title)

        plt.show()

    def _normalize_for_display(self, band_data):
        band_data = reshape_as_image(np.array(band_data))
        lower_perc = np.percentile(band_data, 2, axis=(0,1))
        upper_perc = np.percentile(band_data, 98, axis=(0,1))
        return (band_data - lower_perc) / (upper_perc - lower_perc)

