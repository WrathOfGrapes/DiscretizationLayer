import numpy as np
import pandas as pd
import time

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from net import make_net
from make_data import make_data

random_state = 13
np.random.seed(random_state)

print 'read data'
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
test_ID = df_test['ID_code'].values
Y = df_train.target.values.astype(np.float32)
df_train = df_train.drop(['ID_code','target'], axis=1)
df_test = df_test.drop(['ID_code'], axis=1)

data = pd.read_csv('data_train.csv')
X_T = data.drop(columns=['ID_code', 'target']).values

scaler = StandardScaler()
scaler.fit(X_T)

X = pd.concat([df_train, df_test], axis=0, sort=False, ignore_index=True).values
del df_train, df_test

X = scaler.transform(X)

model, local_model = make_net(200, 1e-4)

model.load_weights('model')

X = make_data(local_model, X)

print 'start training of GaussianNB...'
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import QuantileTransformer
start_tiem = time.time()

_X = X[:len(Y)]
Xt = X[len(Y):]
X = _X

clf = make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB())
clf.fit(X, Y)
y_diff = Y - clf.predict_proba(X)[:, 1]
yt_nb = clf.predict_proba(Xt)[:, 1]
Y = y_diff

print 'start training of LightGBM...'

n_predict = 0
valid = np.zeros((len(test_ID),))
for fold_id, (IDX_train, IDX_test) in enumerate(KFold(n_splits=10, random_state=random_state, shuffle=False).split(Y)):
	X_train = X[IDX_train]
	X_test = X[IDX_test]
	Y_train = Y[IDX_train]
	Y_test = Y[IDX_test]

	lgb_params = {
		"objective": "regression",
		"metric": "mse",
		"max_depth": 2,
		"num_leaves": 2,
		"learning_rate": 0.055,
		"bagging_fraction": 0.3,
		"feature_fraction": 0.15,
		"lambda_l1": 5,
		"lambda_l2": 5,
		"bagging_seed": fold_id+random_state,
		"verbosity": -1,
		'num_threads': 16,
		"seed": fold_id+random_state
	}

	lgtrain = lgb.Dataset(X_train, label=Y_train)
	lgtest = lgb.Dataset(X_test, label=Y_test)
	evals_result = {}
	lgb_clf = lgb.train(lgb_params, lgtrain, 35000, valid_sets=[lgtrain, lgtest], early_stopping_rounds=500,
						verbose_eval=2000, evals_result=evals_result)
	valid += lgb_clf.predict(Xt).reshape((-1,))
	n_predict += 1
	if time.time() - start_tiem > 6900:
		break

valid = (valid / n_predict) + yt_nb
valid = np.clip(valid, 0.0, 1.0)
print('save result.')
pd.DataFrame({'ID_code': test_ID, 'target': valid}).to_csv('submission.csv', index=False)
print('done.')


