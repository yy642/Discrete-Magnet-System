"""
In this example, neural networks(NN) for the forward and inverse problem are constructed and trained in tandem.
Forward problem is denoted as X->Y, and inverse problem is denoted as Y->X.
In this specific case, the inverse problem Y->X suffers from nonuniqueness, i.e., the single Y can be mapped
to multiple different X, which casues conflicting training instances and making NN hard to converge. 

To overcome this issue, an "auto-encoder" style NN are constructed and trained:
(1) the NN for forward problem is constructed and trained normally: X->Y 
(2) the NN for inverse problem is constructed: Y->X
(3) the pre-trained forward NN is connected after inverse NN, during the training, the forward NN is frozen:
    Y->(X->Y)
(4) After training the above NN, the middle layer X is extracted out, which is the desired solution to 
    inverse problem.
"""

import os,time,re
import sys
import pickle,gzip
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import numpy.random as random
import itertools

from l2distance import l2distance
from utils import *
from analyze import *
from Buckets import *

from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout
from keras.layers import BatchNormalization, Input, Lambda
from keras import regularizers
from keras.losses import mse, binary_crossentropy
from keras.optimizers import SGD


"""
Generate training data
"""
N=3
M=3
pads = gen_magnet_pads(N,M)
dlist = np.arange(0.5,5,0.25)
#interaction_tensor_2D = gen_2D_pos(N,M)
#interaction_tensor_3D = face_to_face_tensor(interaction_tensor_2D, dlist)
unique_xy, idxlist = initialize_buckets(interaction_tensor_2D)

my_x = []
my_y = []

for i in range(len(pads)):
    for j in range(len(pads)):
        pad1 = pads[i]
        pad2 = pads[j]
        bucket = outerToBucket(np.outer(pad1, pad2), idxlist)
        twopads = np.concatenate((pad1,pad2)) 
        my_x.append(twopads)
        my_y.append(bucket)

dataX = np.array(my_x)
dataY = np.array(my_y)
X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.2)
X_train_AE = X_train.copy()
Y_train_AE = Y_train.copy()
"""
Define the layer dimension, for the forward problem, three non-linear layer followed by 
one linear layer of NN are constructed
"""
layer_dim = [M*M + N*N, M**2*N**2, len(unique_xy)]
"""
Construct and train the forward problem first:
X->Y
"""
forward_model = Sequential()
forward_model.add(Dense(units=layer_dim[0], activation='tanh',input_dim=layer_dim[0]))
forward_model.add(Dense(units=layer_dim[1], activation='tanh'))
forward_model.add(Dense(units=layer_dim[2], activation='tanh'))
forward_model.add(Dense(units=layer_dim[2], activation='linear'))
forward_model.compile(optimizer='sgd',
             loss='mean_squared_error',
             metrics=['mse'],)

num_epochs=500
batch_size=128
forward_model.fit(x=X_train_AE, y=Y_train_AE,
                 epochs=num_epochs,
                 batch_size=batch_size,
                 shuffle=True,
                 validation_split=0.2)
"""
Check the accuracy of forward modeling.
Save the forward model layer so that it can be used later. 
"""
y_pred =forward_model.predict(X_test)
forward_model.save('forward_model.h5')
np.save("y_pred",y_pred)
np.save("y_test",Y_test)

"""
Construct NN for inverse problem, Y->X, and connect the last layer to the first layer of pre-trained 
forward layer, fix the weights of all the pre-trained forward layers.
"""
inverse_model = Sequential()
inverse_model.add(Dense(units=layer_dim[-1], activation='tanh',input_dim=layer_dim[-1]))
inverse_model.add(Dense(units=layer_dim[-2], activation='tanh'))
inverse_model.add(Dense(units=layer_dim[-3], activation='tanh'))
inverse_model.add(forward_model.layers[0])
inverse_model.add(forward_model.layers[1])
inverse_model.add(forward_model.layers[2])
inverse_model.add(forward_model.layers[3])

"""
fix the weights of the pre-trained forward layers
"""
for layer in inverse_model.layers[4:]:
    layer.trainable = False

inverse_model.compile(optimizer='adam',
             loss='mean_squared_error',
             metrics=['mse'])

num_epochs=500
batch_size=128
inverse_model.fit(x=Y_train_AE, y=Y_train_AE,
                 epochs=num_epochs,
                 batch_size=batch_size,
                 shuffle=True,
                 validation_data=(Y_train_AE, Y_train_AE))()
"""
extract the middle layer
"""
get_middle_layer_output = K.function([inverse_model.layers[0].input],
                                  [inverse_model.layers[3].output])
layer_output = get_middle_layer_output([Y_test])[0]
np.save("X_pred",layer_output)
inverse_model.save('inverse_model.h5')
