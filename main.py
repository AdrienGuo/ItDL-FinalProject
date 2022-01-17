import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

directory = "D:\Data\g-research-crypto-forecasting\\"

train_path = directory + "train.csv"
asset_path = directory + "asset_details.csv"



train_data = pd.read_csv(train_path)
print(train_data[:60])
print(train_data[:60].isna().sum())
print(type(train_data))

# +---------------------------------------+
# |   DAE                                 |
# +---------------------------------------+
class DenoisingAE:
    def __init__(self):
        pass

    def gaussian_noise(self, shape):
        mu, sigma = 0, 0.01
        noise = np.random.normal(mu, sigma, shape)
        return noise

    def create_model(self, in_dim, out_dim):
        model = Sequential()

        # encoder
        model.add(Dense(units=5, input_dim=in_dim, activation="relu"))

        # decoder
        model.add(Dense(units=out_dim))

        plot_model(model, to_file='autoencoder.png', show_shapes=True)
        return model

    def train(self, data, label):
        in_dim = data.shape[1]
        out_dim = label.shape[1]
        self.model = self.create_model(in_dim, out_dim)
        self.model.compile(optimizer="adam", loss="mse")
        data = data + self.gaussian_noise(data.shape)               # add gaussian noise to data
        train_history = self.model.fit(x=data, y=label, epochs=10, batch_size=2)
    
    def predict(self, test):
        predictions = self.model.predict(test)
        print(predictions)


# get index of row which contains nan
rows_with_nan = [index for index, row in train_data[:60].iterrows() if row.isnull().any()]
print(rows_with_nan)

# get df includes/excludes nan
train_data_with_nan = train_data[:60].iloc[[i for i in range(train_data[:60].shape[0]) if i in rows_with_nan], :]
train_data_wo_nan = train_data[:60].iloc[[i for i in range(train_data[:60].shape[0]) if i not in rows_with_nan], :]
print(train_data_wo_nan)

train_data_with_nan = train_data_with_nan.fillna(0)

DAE = DenoisingAE()
DAE.train(train_data_wo_nan, train_data_wo_nan)
DAE.predict(train_data_with_nan)



