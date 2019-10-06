import os
import numpy as np
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt
import keras
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Dropout, MaxPooling1D, Flatten, GlobalAveragePooling1D, Reshape
from keras.regularizers import l2 
from collections import Counter

print('\n')
print("---   Tempus Case Study    ---")
print("--- Michael Richard Wells  ---")


# LOAD DATA 
print('\n')
print("Loading Data ...")
data = np.genfromtxt(fname = "/home/mwe11s/Desktop/DScasestudy.txt",dtype = 'int')
print("... Data Loaded")
print('\n')
# Pre Processing

row = data.shape[0] - 1 # 530 
col = data.shape[1] - 1 # 16562

row, col = data.shape # 531 x 16563
print("row:",  row)
print("col:",  col)
print('\n')
Y = data[:,0]
X = data[:,1:(col-1)]

print(Y[0])
Y[0] = 1
print(Y[0])

# Test Set will contain 424 Samples ~80%
# Train Set will contain 107 Samples ~20%

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=.2)
n_timesteps, n_features, n_outputs = x_train.shape[0], x_train.shape[1], y_train.shape[0]


# Build Model 
model = Sequential()
model.add(Dense(1000, activation = 'relu', input_dim = n_features, kernel_regularizer = l2(0.01)))
model.add(Dropout(.3, noise_shape = None, seed = None))
model.add(Dense(100, activation = 'relu', kernel_regularizer = l2(0.01)))
model.add(Dropout(.3, noise_shape = None, seed = None))
model.add(Dense(1,activation = 'sigmoid'))


# Run Model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model_output = model.fit(x_train, y_train, epochs=25, batch_size=128, verbose=1, validation_data=(x_test, y_test),)


# Evaluate Performence 


y_pred = model.predict(x_test)
rounded = [round(x[0]) for x in y_pred]
y_pred1 = np.array(rounded, dtype = 'int64')

print("\n")
print('Training Accuracy: ', np.mean(model_output.history["accuracy"]))
print('Validation Accuracy: ', np.mean(model_output.history["val_accuracy"]))
print("\n")

print("\n")
print(confusion_matrix(y_test, y_pred1))
print("\n")
print(precision_score(y_test, y_pred1))
print("\n")

print("\n")
print(Counter(y_train).keys()) # equals to list(set(words))
print(Counter(y_train).values()) # counts the elements' frequency
print("\n")

print("\n")
print(Counter(y_test).keys()) # equals to list(set(words))
print(Counter(y_test).values()) # counts the elements' frequency
print("\n")


# Plot accuracy 
plt.plot(model_output.history['accuracy'])
plt.plot(model_output.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss
plt.plot(model_output.history['loss'])
plt.plot(model_output.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()