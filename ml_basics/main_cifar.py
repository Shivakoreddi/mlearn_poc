# Importing the necessary modules
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.initializers import glorot_normal
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar10

# Preparing the dataset
(X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw) = cifar10.load_data()
X_train = X_train_raw.reshape(X_train_raw.shape[0], -1) / 255
X_test_all = X_test_raw.reshape(X_test_raw.shape[0], -1) / 255
X_validation, X_test = np.split(X_test_all, 2)
Y_train = to_categorical(Y_train_raw)
Y_validation, Y_test = np.split(to_categorical(Y_test_raw), 2)

# Build the neural network using Keras
model = Sequential()
model.add(Dense(1200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(500, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

# Compiling the model of our neural network
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Fitting the model to check its performance
history = model.fit(X_train, Y_train,
                    validation_data=(X_validation, Y_validation),
                    epochs=5, batch_size=32)