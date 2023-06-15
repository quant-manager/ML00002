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

###############################################################################
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # drop NUMA warnings from TF
import numpy as np
import tensorflow as tf

###############################################################################
# An attempt to use TF 2.12.3 did not work (error about libdevice.10.bc).
# See this link for details:
# https://github.com/tensorflow/tensorflow/issues/58681
# The simplest solution was to downgrade to TF 2.9.1, which works.
print("Tensorflow version: {version}".format(version=tf.__version__))

###############################################################################
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


###############################################################################
# Step 1: Download and unzip 2 files to ./data
#
# https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_train.csv
# https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_test.csv


###############################################################################
# Step 2: load dataset

print("Loading training set... ", end="")
fname_train = os.path.join('.', 'data','mnist_train.csv')
if os.path.isfile(fname_train) :
    data_train = np.loadtxt(
        fname = fname_train,
        delimiter = ',',
        skiprows = 1,)
else :
    print(
        "EOFError: missing training set CSV file.\n" +
        "Download and unzip training dataset to './data' from\n" +
        "https://www.kaggle.com/datasets/oddrationale/" +
        "mnist-in-csv?resource=download&select=mnist_train.csv")
    sys.exit(-1)
print("Done.")

print("Loading testing set... ", end="")
fname_test = os.path.join('.', 'data','mnist_test.csv')
if os.path.isfile(fname_test) :
    data_test = np.loadtxt(
        fname = fname_test,
        delimiter = ',',
        skiprows = 1,)
else :
    print(
        "EOFError: missing testing set CSV file.\n" +
        "Download and unzip testing dataset to './data' from\n" +
        "https://www.kaggle.com/datasets/oddrationale/" +
        "mnist-in-csv?resource=download&select=mnist_test.csv")
    sys.exit(-1)
print("Done.")
print()


###############################################################################
# Step 3: extract independent and dependent variables
print("Extracting independent and dependent variables... ", end="")

x_train = data_train[:, 1:]
y_train = data_train[:, 0]
print("Done.")
print("Training set: {n_examples:d} examples, {n_features:d} features".format(
    n_examples = x_train.shape[0], n_features = x_train.shape[1]))
assert(len(y_train.shape) == 1)
assert(y_train.shape[0] == x_train.shape[0])

x_test = data_test[:, 1:]
y_test = data_test[:, 0]
print("Testing set: {n_examples:d} examples, {n_features:d} features".format(
    n_examples = x_test.shape[0], n_features = x_test.shape[1]))
assert(len(y_test.shape) == 1)
assert(y_test.shape[0] == x_test.shape[0])
print()


###############################################################################
# Step 4: prepare and transform the data
print("Normalizing independent variables... ", end="")
x_train = x_train / 255.0
x_test = x_test / 255.0
assert(np.all(x_train <= 1.0) and np.all(x_train >= 0.0))
assert(np.all(x_test <= 1.0) and np.all(x_test >= 0.0))
print("Done.")
print("Dependent variable classes: ", ", ".join(
    ["{:.0f}".format(val) for val in np.unique(y_train)]))
print()


###############################################################################
# Step 5: instanciate a model

print("Instantiating model... ", end="")
model = tf.keras.models.Sequential(
[
    tf.keras.layers.Dense(units = 16, activation = 'relu'),
    tf.keras.layers.Dense(units = 10, activation = 'softmax'),
])
print("Done.")
print()


###############################################################################
# Step 6: compile the model

print("Compiling model... ", end="")
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
    )
print("Done.")
print()


###############################################################################
# Step 7: fit the model

print("Fitting model... ", end="")
model.fit(
    x = x_train,
    y = y_train,
    batch_size = 16,
    epochs = 5,
    verbose = 2,
    validation_split = 0.1,
    shuffle = True,
    )
print("Done.")
print()
print("Model Summary:")
print(model.summary())
print()


###############################################################################
# Step 8: evaluate the model on training set

print("Evaluating model on training set... ", end="")
lst_metrics_train = model.evaluate(
    x = x_train,
    y = y_train,
    verbose=0,
    )
print("Done.")
print("Training set metrics:")
for (metric_index, metric_name) in enumerate(model.metrics_names) :
    print("{metric_name} = {metric_value:.4f}".format(
        metric_name = metric_name,
        metric_value = lst_metrics_train[metric_index]))
print()


###############################################################################
# Step 9: evaluate the model on testing set

print("Evaluating model on testing set... ", end="")
lst_metrics_test = model.evaluate(
    x = x_test,
    y = y_test,
    verbose=0,
    )
print("Done.")
print("Testing set metrics:")
for (metric_index, metric_name) in enumerate(model.metrics_names) :
    print("{metric_name} = {metric_value:.4f}".format(
        metric_name = metric_name,
        metric_value = lst_metrics_test[metric_index]))
print()
