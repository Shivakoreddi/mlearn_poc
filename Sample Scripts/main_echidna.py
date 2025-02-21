# importing the necessary modules
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
import echidna as data
import boundary

# Data pre-processing
X_train = data.X_train
X_validation = data.X_validation
Y_train = to_categorical(data.Y_train)
Y_validation = to_categorical(data.Y_validation)

# Building neural network model
model = Sequential()
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

# Calling the training function for our neural network
model.fit(X_train, Y_train,
          validation_data=(X_validation, Y_validation),
          epochs=1000, batch_size=25)

# Display the descision boundary
boundary.show(model, data.X_train, data.Y_train)