import numpy as np
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt
import keras
import sklearn
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Dropout, MaxPooling1D, Flatten, GlobalAveragePooling1D, Reshape
from keras.regularizers import l2 

print("---   Tempus Case Study    ---")
print("--- Michael Richard Wells  ---")

print("Loading Data ...")
# LOAD DATA 
data = np.genfromtxt(fname = "/home/michael/Desktop/DScasestudy.txt",dtype = 'int')

# Pre Processing
print(type(data)) 
print("... Data Loaded")


row = data.shape[0] - 1 # 530 
col = data.shape[1] - 1 # 16562

row, col = data.shape # 531 x 16563
print("row:",  row)
print("col:",  col)

Y = data[:,0]
X = data[:,1:(col-1)]

# 2 Groups 1 and 0
# 0:123 = 1 
# 124:530 = 0
# print(Y)


# Test Set will contain 424 Samples ~80%
# Train Set will contain 107 Samples ~20%
# I want to take 80% from each group to make sure I am not bias to one group 
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=.2)

n_timesteps, n_features, n_outputs = x_train.shape[0], x_train.shape[1], y_train.shape[0]

print(n_timesteps)
print(n_features)
print(n_outputs)

model = Sequential()
# layer 1 
model.add(Dense(100, activation = 'relu', input_dim = n_features, kernel_regularizer = l2(0.01)))
model.add(Dropout(.3, noise_shape = None, seed = None))
# layer 2 
model.add(Dense(100, activation = 'relu', kernel_regularizer = l2(0.01)))
model.add(Dropout(.3, noise_shape = None, seed = None))

# Output layer 
model.add(Dense(1,activation = 'sigmoid'))


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model_output = model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=1, validation_data=(x_test, y_test),)

print('Training Accuracy: ', np.mean(model_output.history["acc"]))
print('Validation Accuracy: ', np.mean(model_output.history["val_acc"]))



y_pred = model.predict(x_test)
rounded = [round(x[0]) for x in y_pred]
y_pred1 = np.array(rounded, dtype = 'int64')

print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))





# Plot training & validation accuracy values
plt.plot(model_output.history['acc'])
plt.plot(model_output.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(model_output.history['loss'])
plt.plot(model_output.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()