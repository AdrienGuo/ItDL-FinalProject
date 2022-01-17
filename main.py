import pandas as pd
import numpy as np

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import GaussianNoise
from keras.layers import Dense
from keras.utils import plot_model

directory = "D:\Data\g-research-crypto-forecasting\\"

train_path = directory + "train.csv"
asset_path = directory + "asset_details.csv"

train_data = pd.read_csv(train_path)

# Split dataframe by Asset_ID
def split_dataframe(df):
    df_list = []
    for i in range(14):
        df_list.append(df[df['Asset_ID']==i])
    return df_list

# This is a type(list)
df_ori = split_dataframe(train_data)
df_list = df_ori

# Normalization
def normalization(df):
    norm_Count = preprocessing.MinMaxScaler()
    norm_Open = preprocessing.MinMaxScaler()
    norm_High = preprocessing.MinMaxScaler()
    norm_Low = preprocessing.MinMaxScaler()
    norm_Close = preprocessing.MinMaxScaler()
    norm_Volume = preprocessing.MinMaxScaler()
    norm_VWAP = preprocessing.MinMaxScaler()

    df[['Count']] = norm_Count.fit_transform(df[['Count']])
    df[['Open']] = norm_Open.fit_transform(df[['Open']])
    df[['High']] = norm_High.fit_transform(df[['High']])
    df[['Low']] = norm_Low.fit_transform(df[['Low']])
    df[['Close']] = norm_Close.fit_transform(df[['Close']])
    df[['Volume']] = norm_Volume.fit_transform(df[['Volume']])
    # df[['VWAP']] = norm_VWAP.fit_transform(df[['VWAP']])

    dict = {
        "Count": norm_Count,
        "Open": norm_Open,
        "High": norm_High,
        "Low": norm_Low,
        "Close": norm_Close,
        "Volume": norm_Volume,
        "VWAP": norm_VWAP,
    }

    return df, dict

# Do Norm
for i in range(len(df_list)):
    dict_list = []
    df, dict = normalization(df_list[i])
    df_list[i] = df
    dict_list.append(dict)

print(df_list[11])


# Split dataframe by nan
def split_nan(df):
    nan_list = []
    for idx in range(df.shape[0]):
        if df.iloc[idx].isnull().any():
            nan_list.append(idx)
    # rows_with_nan = [index for index, row in df.iterrows() if row.isnull().any()]
    # df_with_nan = df.iloc[[index for index, row in df.iterrows() if index in rows_with_nan], :]
    # df_wo_nan = df.iloc[[index for index, row in df.iterrows() if index not in rows_with_nan], :]
    df_with_nan = df.iloc[[i for i in range(df.shape[0]) if i in nan_list], :]
    df_wo_nan = df.iloc[[i for i in range(df.shape[0]) if i not in nan_list], :]
    df_with_nan = df_with_nan.reset_index(drop=True)
    df_wo_nan = df_wo_nan.reset_index(drop=True)
    return df_with_nan, df_wo_nan

df_with_nan, df_wo_nan = split_nan(df_list[11][:60])
print(df_with_nan)
print(df_wo_nan)


# +---------------------------------------+
# |   DAE                                 
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

        # GaussianNoise
        model.add(GaussianNoise(0.01, input_shape=(in_dim, )))

        # Encoder
        model.add(Dense(units=5, activation="relu"))

        # Decoder
        model.add(Dense(units=out_dim))

        plot_model(model, to_file='Denoising_AutoEncoder.png', show_shapes=True)
        return model

    def train(self, data, label):
        in_dim = data.shape[1]
        out_dim = label.shape[1]
        self.model = self.create_model(in_dim, out_dim)
        self.model.compile(optimizer="adam", loss="mse")
        # data = data + self.gaussian_noise(data.shape)               # Add gaussian noise to data
        train_history = self.model.fit(x=data, y=label, epochs=10, batch_size=2)
    
    def predict(self, test):
        predictions = self.model.predict(test)
        return predictions

df_with_nan = df_with_nan.fillna(0)
DAE = DenoisingAE()
DAE.train(df_wo_nan, df_wo_nan)
predictions = DAE.predict(df_with_nan)

def add_predictions(df_with_nan, predictions):
    pred_list = [predictions[i][-1] for i in range(predictions.shape[0])]
    pred_df = pd.DataFrame(pred_list, columns=["Target"])

    # Remove column "Target"
    df_with_nan.drop("Target", axis=1, inplace=True)

    concat_df = pd.concat([df_with_nan, pred_df], axis=1)
    return concat_df

df_with_nan = add_predictions(df_with_nan, predictions)

def merge(df_with_nan, df_wo_nan):
    concat_df = pd.concat([df_with_nan, df_wo_nan])
    df = concat_df.sort_values(by=['timestamp'])
    df = df.reset_index(drop=True)
    return df

df_all = merge(df_with_nan, df_wo_nan)
print(df_all)
