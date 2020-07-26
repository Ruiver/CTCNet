"""
author: ouyangtianxiong
date:2019/12/24
des: this package implements some baseline
"""
import sys
sys.path.append('../')
from sklearn import  svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_set.deap_feature import DEAP

def get_svm_clf():
    return svm.SVC()

def get_rf_clf():
    return RandomForestClassifier(n_estimators=512, n_jobs=6)

def get_ab_clf():
    return AdaBoostClassifier(n_estimators=512)
def testing(model, test_X, test_Y):
    predict = model.predict(test_X)
    print("classification report:\n{}".format(classification_report(test_Y, predict)))
    print("accuracy score:\n{}".format(accuracy_score(test_Y, predict)))
    print("confusion matrix:\n{}".format(confusion_matrix(test_Y, predict)))


if __name__ == '__main__':
    individual = 11
    nor_method = 0
    target = 4
    data = DEAP(individual, nor_method)
    train_X, train_Y = data.get_train_data()
    test_X, test_Y = data.get_test_data()
    model = get_rf_clf()
    model.fit(train_X, train_Y[:, target])
    testing(model, test_X, test_Y[:, target])
