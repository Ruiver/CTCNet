"""
author: ouyangtianxiong
date:2019/12/26
des:implement a baseline using xgbboost for classification
"""
import sys
sys.path.append('../')
import os
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from data_set.seed_iv import SEED_IV
from data_set.deap_feature import DEAP


# eeg = SEED_IV(session=1, individual=1, modal='concat', shuffle=False, balance=True,
#                       normalization=1)
eeg = DEAP()

train_X, train_Y = eeg.get_train_data()
test_X, test_Y = eeg.get_test_data()
train_Y = train_Y[:, -1]
test_Y = test_Y[:, -1]
dtrain = xgb.DMatrix(train_X, train_Y)
dtest = xgb.DMatrix(test_X)

param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softmax'}

evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 4
model = xgb.XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=160, silent=True, objective='multi:softmax')

model.fit(train_X, train_Y)
plot_importance(model)
y_pred = model.predict(test_X)
print("classification report:\n",classification_report(test_Y, y_pred))
print("confusion matrix:\n", confusion_matrix(test_Y, y_pred))
print("accuracy score\n", accuracy_score(test_Y, y_pred))
plot_importance(model)

