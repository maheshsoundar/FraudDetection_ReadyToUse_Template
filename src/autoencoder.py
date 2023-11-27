import numpy as np
from sklearn.metrics import mean_squared_error

import tensorflow as tf
tf.keras.utils.set_random_seed(5)  # sets seeds for base-python, numpy and tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# create a model by subclassing Model class in tensorflow
class AutoEncoderModel(Model):
  def __init__(self, data_shape):
    super().__init__()
    self.model = self.__create_model(data_shape)

  def __create_model(self,data_shape):
    input = Input(shape=data_shape)
    encoder = Sequential([
      Dense(64, activation='relu'),
      Dropout(0.5),
      Dense(32, activation='relu'),
      Dropout(0.5),
      Dense(16, activation='relu'),
      Dropout(0.5),
      Dense(8, activation='relu')
    ]) (input)
    decoder = Sequential([
      Dense(16, activation='relu'),
      Dropout(0.5),
      Dense(32, activation='relu'),
      Dropout(0.5),
      Dense(64, activation='relu'),
      Dropout(0.5),
      Dense(data_shape, activation='sigmoid')
    ])(encoder)
    model = Model(inputs=input, outputs=decoder)
    model.compile(loss='mean_squared_error', metrics=['mae'], optimizer='adam')
    return model

  def __find_threshold(self,data):
    pred = self.model.predict(data)
    loss = mean_squared_error(data,pred)
    return np.percentile(loss, 98) #setting 98th percentile as the threshold

  def run(self, data,epochs=10,batch_size=256):
    history = self.model.fit(data,data,epochs=epochs,batch_size=batch_size,verbose=1,validation_split=0.1)
    self.threshold = self.__find_threshold(data)
    return history

  def predict(self,test):
    predictions = self.model.predict(test)
    class_labels = []
    for i in range(0,len(test)):
      if(mean_squared_error(test[i],predictions[i]) < self.threshold):
        class_labels.append(0)
      else:
        class_labels.append(1)
    return class_labels

  def get_model(self):
    return self.model

