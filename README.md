# Towards Global Estimation of Ground-Level NO2 Pollution with Deep Learning and Remote Sensing

This repository provides the code for a forthcoming paper on the estimation of surface-level NO2 concentrations from various data sources. 

## Local Model
The local model (`local_model` directory) estimates surface NO2 for locations of EEA air quality ground stations based on station metadata with XGBoost.

## OSM Model
This model (`osm_model` directory) is based on land cover statistics derived from the OpenStreetMap project and estimates NO2 with XGBoost.

## Satellite Model
The satellite model (`satellite_model` directory) uses remote sensing data (Sentinel-1 and Sentinel-5P) to expand high-resolution NO2 estimation to a global scale.
The model is an artificial neural network with two input streams for the different data modalities, trained against ground-stations measurements in Europe.
The remote sensing data for training the network has to be downloaded separately from zenodo.com
