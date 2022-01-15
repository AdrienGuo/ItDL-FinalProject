import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import plot_model

directory = "D:\Data\g-research-crypto-forecasting\\"

train_path = directory + "train.csv"
asset_path = directory + "asset_details.csv"

train_data = pd.read_csv(train_path)
print(train_data[-750400:])
print(train_data[-750400:].isna().sum())

encoder = Sequential([Dense(2, input_shape=[3])])
decoder = Sequential([Dense(3, input_shape=[2])])
autoencoder = Sequential([encoder, decoder])
autoencoder.compile(loss="mse", optimizer=SGD(lr=0.01))
plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)
