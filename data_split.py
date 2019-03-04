import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')

indicies_train, indicies_test = train_test_split(range(len(data)), train_size=0.8)


data_train = data.iloc[indicies_train]
data_test = data.iloc[indicies_test]

data_train.to_csv('data_train.csv', index=False)
data_test.to_csv('data_test.csv', index=False)