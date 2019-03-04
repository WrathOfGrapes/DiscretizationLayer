import pandas as pd
import numpy as np


data = pd.read_csv('train.csv')
X = data.drop(columns=['ID_code', 'target']).values
y = data['target'].values

preds = pd.read_csv('prediction_net.csv')
preds = preds.drop(columns=['ID_code']).values

errors = np.abs(preds[:, 0] - preds[:, 1])

print np.mean(errors > 0.5)