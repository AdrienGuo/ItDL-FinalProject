from re import sub
import pandas as pd
import numpy as np
import time

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import GaussianNoise
from keras.layers import Dense
# from keras.utils import plot_model

directory = "D:\Data\ItDL_FinalProject\g-research-crypto-forecasting\\"

train_path = directory + "train.csv"
asset_path = directory + "asset_details.csv"

# Drop rows for VWAP column contain nan
train_data = pd.read_csv(train_path)
train_data.dropna(subset=["VWAP"], inplace=True)
print(train_data.isnull().sum())

# Split dataframe by Asset_ID
def split_dataframe(df):
    df_list = []
    for i in range(14):
        df_list.append(df[df['Asset_ID']==i])
    return df_list

# This is a type(list)
df_ori = split_dataframe(train_data)
df_list = df_ori

# Find nan index
def find_nan(df):
    df = df.to_numpy()
    index_nan = np.where(np.isnan(df))
    return index_nan

# print("Start find nan")
# index_nan_list = []
# for i in range(14):
#     index_nan = find_nan(df_list[i])
#     index_nan_list.append(index_nan)
# print("End find nan")
# print(index_nan_list[0])

start = time.time()
nan_index = find_nan(df_list[11][:])[0]
# This will return two array [[rows], [columns]]
# But I only want the row index
end = time.time()
print("="*100)
print("Nan index:")
print(nan_index)
print("="*100)
print("Time of find_nan: {}".format(end-start))

# Normalization
def normalization(df):
    norm_Count = preprocessing.MinMaxScaler()
    norm_Open = preprocessing.MinMaxScaler()
    norm_High = preprocessing.MinMaxScaler()
    norm_Low = preprocessing.MinMaxScaler()
    norm_Close = preprocessing.MinMaxScaler()
    norm_Volume = preprocessing.MinMaxScaler()
    norm_VWAP = preprocessing.MinMaxScaler()
    norm_Target = preprocessing.MinMaxScaler()

    df[['Count']] = norm_Count.fit_transform(df[['Count']])
    df[['Open']] = norm_Open.fit_transform(df[['Open']])
    df[['High']] = norm_High.fit_transform(df[['High']])
    df[['Low']] = norm_Low.fit_transform(df[['Low']])
    df[['Close']] = norm_Close.fit_transform(df[['Close']])
    df[['Volume']] = norm_Volume.fit_transform(df[['Volume']])
    df[['VWAP']] = norm_VWAP.fit_transform(df[['VWAP']])
    df[['Target']] = norm_Target.fit_transform(df[['Target']])

    dict = {
        "Count": norm_Count,
        "Open": norm_Open,
        "High": norm_High,
        "Low": norm_Low,
        "Close": norm_Close,
        "Volume": norm_Volume,
        "VWAP": norm_VWAP,
        "Target": norm_Target,
    }

    return df, dict

def normalization_nan(df, dict):
    df[['Count']] = dict["Count"].fit_transform(df[['Count']])
    df[['Open']] = dict["Open"].fit_transform(df[['Open']])
    df[['High']] = dict["High"].fit_transform(df[['High']])
    df[['Low']] = dict["Low"].fit_transform(df[['Low']])
    df[['Close']] = dict["Close"].fit_transform(df[['Close']])
    df[['Volume']] = dict["Volume"].fit_transform(df[['Volume']])
    df[['VWAP']] = dict["VWAP"].fit_transform(df[['VWAP']])
    return df

# Split dataframe by nan
def split_nan(df):
    nan_list = []
    for idx in range(df.shape[0]):
        if df.iloc[idx].isnull().any():
            nan_list.append(idx)
    print("Here is nan_index: {}".format(nan_list))
    # rows_with_nan = [index for index, row in df.iterrows() if row.isnull().any()]
    # df_with_nan = df.iloc[[index for index, row in df.iterrows() if index in rows_with_nan], :]
    # df_wo_nan = df.iloc[[index for index, row in df.iterrows() if index not in rows_with_nan], :]
    df_with_nan = df.iloc[[i for i in range(df.shape[0]) if i in nan_list], :]
    df_with_nan = df_with_nan.reset_index(drop=True)
    df_with_nan = df_with_nan.fillna(0)
    df_wo_nan = df.iloc[[i for i in range(df.shape[0]) if i not in nan_list], :]
    df_wo_nan = df_wo_nan.reset_index(drop=True)
    return df_with_nan, df_wo_nan

start = time.time()
df_with_nan, df_wo_nan = split_nan(df_list[11][:60])
end = time.time()
print("Time for split_nan: {}".format(end-start))
print("="*100)
print("Dataframe contains nan:")
print(df_with_nan)

print("="*100)
print("Dataframe without nan:")
print(df_wo_nan)

# Do Norm
df_wo_nan, norm_dict = normalization(df_wo_nan)
print("\n"+"="*100)
print("Do Normalization on dataframe without nan:")
print(df_wo_nan)

# Do Norm on df with nan
df_with_nan = normalization_nan(df_with_nan, norm_dict)
print("\n"+"="*100)
print("Do Normalization on dataframe with nan:")
print(df_with_nan)

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

        # plot_model(model, to_file='Denoising_AutoEncoder.png', show_shapes=True)
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
