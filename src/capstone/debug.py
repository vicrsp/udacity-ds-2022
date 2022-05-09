# %% [markdown]
# # LSTM Autoencoder 
# ## Tennessee Eastman Process Simulation Dataset
# 
# Source: https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset

# %%
import pandas as pd
import numpy as np
from model.autoencoder import LSTMAutoencoder, LSTMAutoencoderTransformer
from data.process import TEPDataLoader
from data.utils import get_missing_values_table

import plotly.graph_objects as go
import plotly.express as px 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import set_config
set_config(display="diagram")

# from pandarallel import pandarallel

# pandarallel.initialize(progress_bar=True)

# %% [markdown]
# ## Exploratory Data Analysis
# ### Loading datasets
# 
# A total of 4 datasets are available, already split into train and test. From the data source, the following complete description is available: 
# 
# * Each dataframe contains 55 columns:
# 
#   * Column 1 ('faultNumber') ranges from 1 to 20 in the “Faulty” datasets and represents the fault type in the TEP. The “FaultFree” datasets only contain fault 0 (i.e. normal operating conditions).
# 
#   * Column 2 ('simulationRun') ranges from 1 to 500 and represents a different random number generator state from which a full TEP dataset was generated (Note: the actual seeds used to generate training and testing datasets were non-overlapping).
# 
#   * Column 3 ('sample') ranges either from 1 to 500 (“Training” datasets) or 1 to 960 (“Testing” datasets). The TEP variables (columns 4 to 55) were sampled every 3 minutes for a total duration of 25 hours and 48 hours respectively. Note that the faults were introduced 1 and 8 hours into the Faulty Training and Faulty Testing datasets, respectively.
# 
# * Columns 4 to 55 contain the process variables; the column names retain the original variable names.

# %%
# create the loader class instance
loader = TEPDataLoader('/mnt/d/udacity-ds-2022/src/capstone/data/raw')

# load the training and test datasets
X_train_normal = loader._load('fault_free_training.csv')
# X_test_faulty, X_test_normal = loader.load_testing_datasets()

# %% [markdown]
# ## Modelling

# %% [markdown]
# ### Batch-wise modelling

# %%
# Measured process data in the multiphase batch process are usually stored in the form of a three-dimensional cube
# I×J×K, recording K measured points with J process variables of all the I batches
I = X_train_normal.simulationRun.unique().shape[0]
K = X_train_normal['sample'].unique().shape[0]
J = len(X_train_normal.columns[3:])
print(f'I: {I}, J: {J}, K: {K}')

X_sc = LSTMAutoencoderTransformer.created_windowed_dataset(X_train_normal.iloc[:,3:].values, K, 5)
X_sc_r = LSTMAutoencoderTransformer.reverse_windowed_dataset(X_sc, window=5)  

# %% [markdown]
# ### Training

# %%
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scale', RobustScaler()),
                ('reshape', LSTMAutoencoderTransformer(batch_size=K, window=5)), 
                ('encoder', LSTMAutoencoder(epochs=5, batch_size=128, validation_split=0.2))])

pv_columns = X_train_normal.columns[3:]
X_train = X_train_normal.iloc[:,3:].values
pipe.fit(X_train, None)
# %%

# 


