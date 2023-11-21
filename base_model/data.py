from __future__ import absolute_import
from keras.models import model_from_json
import numpy as np
import os
from scipy.io import loadmat
import hdf5storage as h5
import tables
import json

from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD

def load_model(defFile, weightFile):
    
    # as per http://keras.io/faq/#how-can-i-save-a-keras-model

    # model = model_from_json(open(defFile).read())

    # Load JSON configuration from the file
    with open(defFile, 'r') as json_file:
        json_config = json.load(json_file)

    model = Sequential()

    n_conv = 0

    for layer_config in json_config["layers"]:
        layer_name = layer_config["name"]

        if layer_name == "Convolution2D":

            if n_conv == 0:

                model.add(Conv2D(
                    filters=layer_config["nb_filter"],
                    kernel_size=(layer_config["nb_col"], layer_config["nb_row"]),
                    strides=(layer_config["subsample"][0], layer_config["subsample"][1]),
                    padding=layer_config["border_mode"],
                    activation=layer_config["activation"],
                    input_shape= (layer_config["input_shape"][0], layer_config["input_shape"][1], layer_config["input_shape"][2])
                ))

                n_conv += 1
            
            else:

                model.add(Conv2D(
                    filters=layer_config["nb_filter"],
                    kernel_size=(layer_config["nb_col"], layer_config["nb_row"]),
                    strides=(layer_config["subsample"][0], layer_config["subsample"][1]),
                    padding=layer_config["border_mode"],
                    activation=layer_config["activation"],
                ))
            
        elif layer_name == "Activation":
            model.add(Activation(layer_config["activation"]))
        elif layer_name == "MaxPooling2D":
            model.add(MaxPooling2D(
                pool_size=(layer_config["pool_size"][0], layer_config["pool_size"][1]),
                strides=(layer_config["strides"][0], layer_config["strides"][1]),
                padding=layer_config["border_mode"]
            ))
        elif layer_name == "Flatten":
            model.add(Flatten())
        elif layer_name == "Dense":
            model.add(Dense(
                units=layer_config["output_dim"],
                activation=layer_config["activation"]
            ))
        elif layer_name == "Dropout":
            model.add(Dropout(layer_config["p"]))

    # Compile the model
    optimizer_config = json_config["optimizer"]
    optimizer = SGD(
        learning_rate=optimizer_config["learning_rate"],
        momentum=optimizer_config["momentum"],
        nesterov=optimizer_config["nesterov"]
    )

    model.compile(loss=json_config["loss"], optimizer=optimizer, metrics=["accuracy"])

    # Print the model summary
    model.summary()

    print(weightFile)

    model.load_weights(weightFile, by_name=True)

    return model


def cellstr_from_tables(f):

    data = []
    for entity in f:
        data.append(''.join([chr(x[0]) for x in entity[0]]))

    return data

def load_mat_batch(filePath, source='mat'):

    # Labels may or may not exist
    labels = None
    labelNames = None

    with tables.open_file(filePath) as f:
        if 'nChunk' in f.root._v_leaves.keys():
            nChunk = f.root.nChunk.read()
            data = f.root.features_1.read()
            for ch in range(2, nChunk+1):
                feaName = 'features_{}'.format(ch)
                data = np.append(data, f.root._v_leaves[feaName].read(), axis=0)

            if 'labels_1' in f.root._v_leaves.keys():
                labels = f.root.labels_1.read().flatten()
                for ch in range(2, nChunk+1):
                    labName = 'labels_{}'.format(ch)
                    labels = np.append(labels, f.root._v_leaves[labName].read().flatten(), axis=0)

        else:
            data = f.root.features.read()
            if 'labels' in f.root._v_leaves.keys():
                labels = f.root.labels.read().flatten()

        if source == 'mat':
            data = np.transpose(data)
            data = np.rollaxis(data, 3)

        if 'labelNames' in f.root._v_leaves.keys():
            labelNames = cellstr_from_tables(f.root.labelNames.read())

    return data, labels, labelNames

def load_mat_chunk(filePath,chunkNum=None):

    if chunkNum is None:
        
        try:
            nChunk = h5.loadmat(filePath, variable_names=['nChunk'])['nChunk']
        except:
            nChunk = None

        try:
            struct = h5.loadmat(filePath, variable_names=['labelNames'])
            labelNames = [f[0][0] for f in struct['labelNames'][0]]
        except:
            labelNames = None

        return nChunk, labelNames

    var = 'features_'+str(chunkNum)
    data = h5.loadmat(filePath, variable_names=[var])[var]
    data = np.rollaxis(data, 3)

    try:
        var = 'labels_'+str(chunkNum)
        labels = h5.loadmat(filePath, variable_names=[var])[var]
    except:
        labels = None

    var = 'features'
    data = h5.loadmat(filePath, variable_names=[var])[var]
    data = np.rollaxis(data, 3)

    # data = h5.loadmat(filePath, variable_names=['features'])['features']
    # labels = h5.loadmat(filePath, variable_names=['labels'])['labels']

    return data, labels


