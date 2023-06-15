#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 James James Johnson. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
@author: James J Johnson
@url: https://www.linkedin.com/in/james-james-johnson/
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # drop NUMA warnings from TF
import numpy as np
import tensorflow as tf

# An attempt to use TF 2.12.3 did not work (error about libdevice.10.bc).
# See this link for details:
# https://github.com/tensorflow/tensorflow/issues/58681
# The simplest solution was to downgrade to TF 2.9.1, which works works.
print(tf.__version__)

# Report CPU/GPU availability
print()
print("Fitting will be using {int_cpu_count:d} CPU(s).".format(
    int_cpu_count = len(tf.config.list_physical_devices('CPU'))))
print("Fitting will be using {int_gpu_count:d} GPU(s).".format(
    int_gpu_count = len(tf.config.list_physical_devices('GPU'))))
print()

###############################################################################
# Sequential neural network for multiclass classification
###############################################################################

# Step 1: Download and unzip 2 files to ./data
#
# https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_train.csv
# https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_test.csv

# Step 2: load dataset
data_train = np.loadtxt(
    fname = os.path.join('.', 'data','mnist_train.csv'),
    delimiter = ',',
    skiprows = 1,
    )
data_test = np.loadtxt(
    fname = os.path.join('.', 'data','mnist_test.csv'),
    delimiter = ',',
    skiprows = 1,
    )

# Step 3: split dataset into independednt and dependent variables
x_train = data_train[:, 1:]
y_train = data_train[:, 0]
print(x_train.shape)
print(y_train.shape)
x_test = data_test[:, 1:]
y_test = data_test[:, 0]
print(x_test.shape)
print(y_test.shape)

# Step 4: prepare and transform the data
x_train = x_train / 255.0
print(np.unique(y_train))

# Step 5: design a model
model = tf.keras.models.Sequential(
[
    tf.keras.layers.Dense(units = 16, activation = 'relu'),
    tf.keras.layers.Dense(units = 10, activation = 'softmax'),
])

# Step 6: compile the model
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
    )

# Step 7: fit the model
model.fit(
    x = x_train,
    y = y_train,
    batch_size = 16,
    epochs = 10,
    verbose = 'auto',
    validation_split = 0.1,
    shuffle = True,
    )

# Step 8: evaluate the model on training set
lst_metrics_train = model.evaluate(
    x = x_train,
    y = y_train,
    verbose='auto',
    )

# Step 9: evaluate the model on testing set
lst_metrics_test = model.evaluate(
    x = x_test,
    y = y_test,
    verbose='auto',
    )

print(model.metrics_names)
print(lst_metrics_train)
print(lst_metrics_test)
