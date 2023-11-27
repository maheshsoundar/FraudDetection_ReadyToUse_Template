import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.keras.utils.set_random_seed(5)  # sets seeds for base-python, numpy and tf

class DataUtil:
  def __init__():
    pass

  def load_data(file_dir):
    for _,_,files in os.walk(file_dir):
        for file in files:
            if file.endswith(".csv"):
                filepath = os.path.join(file_dir,file)
                return pd.read_csv(filepath).dropna().drop_duplicates()

  def data_target_split(data,target_col):
    target = data[target_col]
    data.drop(columns=[target_col],inplace=True)
    return data, target

  def normalize_data(data,scaler=None):
    if(scaler is not None):
      return scaler.transform(data),scaler
    scaler = MinMaxScaler().fit(data)
    return scaler.transform(data),scaler