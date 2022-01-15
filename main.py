from gettext import translation
from msilib.schema import Directory
import pandas as pd

directory = "D:\Data\ItDL_FinalProject\g-research-crypto-forecasting\\"

train_path = directory + "train.csv"
asset_path = directory + "asset_details.csv"

train_data = pd.read_csv(train_path)
print(type(train_data))

print(train_data.isna().sum())
