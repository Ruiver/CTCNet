"""
author: ouyangtianxiong
date: 2019/11/20
des: 实现基于单模态特征的SVM分类器
"""
import sys
sys.path.append('../')
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score,accuracy_score
from sklearn.model_selection import  train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler
import numpy as np
import pickle
import os
import codecs
from utils  import plot_confusion_by_heat_map
from pyecharts.charts import HeatMap
# 1 data preparation

class svm_classifier:
    def __init__(self):
        c_range = np.logspace(-10, 10, 11, base=2)
        # gamma_range = np.logspace(-10, 10, 11, base=2)
        # kernel = ['linear', 'rbf']
        parameters = {'C': c_range}
        my_svm = SVC(kernel='linear')
        print("building svm classifier")
        self.svm_clf = Pipeline([('imp1', Imputer(missing_values='NaN', strategy='mean', axis=0)),
                            #('scalar', MinMaxScaler()),
                            ('grid_search', GridSearchCV(my_svm, parameters, n_jobs=-1))])

    def do_fit(self,train_X, train_Y):
        # print(data_dicts)
        self.svm_clf.fit(train_X, train_Y)

    def do_predict(self, test_X, test_Y):
        pred_Y = self.svm_clf.predict(test_X)
        # print("class report \n", classification_report(test_Y, pred_Y))
        # print("confusion matrix \n", confusion_matrix(test_Y, pred_Y))
        # print("accuracy_score \n", accuracy_score(test_Y, pred_Y))
        return classification_report(test_Y,pred_Y), confusion_matrix(test_Y, pred_Y), accuracy_score(test_Y, pred_Y)

    def do_save(self, save_path):
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with codecs.open(save_path, 'wb') as f:
            pickle.dump(self.svm_clf, f)



def read_one_fold(fold):
    """
    read one fold at once
    :param fold:
    :return:
    """
    with codecs.open('../../multimodal_feature_extracted/SEED_IV/%s/fusion_feature.pk' % fold, mode='rb') as f:
        fusion_feature_dicts = pickle.load(f)
    assert isinstance(fusion_feature_dicts, dict), "fusion feature is not type of dict"
    return fusion_feature_dicts

def data_preparation(fold):
    """
    返回SEED_IV中的数据字典，
    :param fold: indicate which fold should use 1 for the first session, 2 for the second session
                 3 for the last session, all for all of them
    :return: data dict
    """
    assert fold in ['1', '2', '3', 'all']
    if fold in ['1', '2', '3']:
        return read_one_fold(fold)
    else:
        d1, d2, d3 = read_one_fold('1'), read_one_fold('2'), read_one_fold('3')
        print(d1.keys())
        print(d2.keys())
        print(d3.keys())
        # assert d1.keys() == d2.keys() and d2.keys() == d3.keys(), 'dict keys not match'
        return {k: np.vstack((d1[k], d2[k], d3[k])) for k in d1.keys()}
def make_train_test_index(fold, flag = "balance"):
    labels_distribute = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                         [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                         [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
    if flag == "balance":
        current_label = labels_distribute[int(fold) - 1]
        label_dict = {k: [] for k in set(current_label)}
        for i, l in enumerate(current_label):
            label_dict[l].append(i)
        train_index = []
        test_index = []
        for k, v in label_dict.items():
            train_index.extend(v[:4])
            test_index.extend(v[4:])
    else:
        idx = list(range(len(labels_distribute[int(fold) - 1])))
        train_index = idx[:17]
        test_index = idx[17:]
    return train_index, test_index
def make_train_test(data,train_index,test_index,modal):
    """
    making training set develop set and testing set
    :param data: original 2-dimension data with label in last row
    :param modal: indicate the feature will be used, in the eeg-eye bimodal eeg means eeg feature, otherwise eye
    :return: shuffled data
    """
    train_data = [data[i] for i in train_index]
    test_data = [data[i] for i in test_index]
    for i, arr in enumerate(train_data):
        if i == 0:
            train = arr
        else:
            train = np.vstack((train, arr))
    for i, arr in enumerate(test_data):
        if i == 0:
            test = arr
        else:
            test = np.vstack((test, arr))
    if modal == 'eeg':
        train_X = train[:, :310]
        train_Y = train[:, -1]
        test_X = test[:, :310]
        test_Y = test[:, -1]
    elif modal == 'eye':
        train_X = train[:, 310:-1]
        train_Y = train[:, -1]
        test_X = test[:, 310:-1]
        test_Y = test[:, -1]
    elif modal == 'concat':
        train_X = train[:, :-1]
        train_Y = train[:, -1]
        test_X = test[:, :-1]
        test_Y = test[:, -1]
    train_Y, test_Y = np.array(train_Y, dtype=np.int), np.array(test_Y, dtype=np.int)
    return train_X, train_Y, test_X, test_Y
# 3 build svm classifier and train
def build_svm(train_X, train_Y, test_X, test_Y, save, model_name):
    """
    svm classifier for Emotion Recognition
    :param train_X:
    :param train_Y:
    :param test_X:
    :param test_Y:
    :param save: save path
    :param model_name: save name
    :return:
    """
    # set the search space of hyperparameter C in svm
    c_range = np.logspace(-10, 10, 30, base=2)
    gamma_range = np.logspace(-10, 10, 11, base=2)
    kernel = ['linear']
    parameters = {'svc__C': c_range}
    print("building svm classifier")
    svm_clf = Pipeline([('imp1', Imputer(missing_values='NaN', strategy='mean', axis=0)),
                        ('scalar', MinMaxScaler()),
                        ('svc', SVC())])
    gs = GridSearchCV(svm_clf, param_grid=parameters, refit=True, cv=10,n_jobs=-1,verbose=0)
    #print(data_dicts)
    gs.fit(train_X, train_Y)
    pred_Y = gs.predict(test_X)
    print("class report \n", classification_report(test_Y, pred_Y))
    print("confusion matrix \n", confusion_matrix(test_Y, pred_Y))
    print("accuracy_score \n", accuracy_score(test_Y, pred_Y))
    print('best params: %r\nbest acc: %.4f' % (gs.best_params_, gs.best_score_))
    # save the best model
    save_path = '../../saved_models/unimodal/' + save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with codecs.open(os.path.join(save_path, model_name), 'wb') as f:
        pickle.dump(gs, f)

def test_B_use_A(fold='1',modal='eeg'):
    """
    use svm model trained in individual A to predict samples of B
    :return:
    """
    session = int(fold)
    model_load_path = '../../saved_models/unimodal/%d' % session
    file_list =  [e for e in os.listdir(model_load_path) if e[-3:] == 'mat']
    data_dicts = data_preparation(str(session))
    train_index, test_index = make_train_test_index(session)
    confusion_mat = np.zeros(shape=(len(file_list), len(file_list)))
    #print(file_list)
    for i, individual in enumerate(file_list):
        control_group = [j for j in range(len(file_list)) if j != i]
        #print(control_group)
        with codecs.open(os.path.join(os.path.join(model_load_path, individual), '%s/best_svm_for_single' % modal if modal in['eeg', 'eye'] else 'best_svm_use_concat'), 'rb') as f:
            svm_clf = pickle.load(f)
        data_self =data_dicts[file_list[i]]
        _, _, X_, Y_ = make_train_test(data_self, train_index, test_index, modal)
        confusion_mat[i][i] = np.round(accuracy_score(Y_, svm_clf.predict(X_)), decimals=3)
        for k in control_group:
            data_individual = file_list[k]
            data = data_dicts[data_individual]
            for idx, arr in enumerate(data):
                if idx==0:
                    train = arr
                else:
                    train = np.vstack((train, arr))
            if modal == 'eeg':
                X = train[:, :310]
            elif modal == 'eye':
                X = train[:, 310:-1]
            elif modal == 'concat':
                X = train[:, :-1]
            Y = train[:, -1]
            Y = np.array(Y, dtype=np.int)
            pred_Y = svm_clf.predict(X)
            #print(i,k)
            confusion_mat[i][k] = np.round(accuracy_score(Y, pred_Y), decimals=3)
    plot_confusion_by_heat_map(confusion_mat * 100, ['P%d' % (i+1) for i in range(len(file_list))], ['P%d' % (i+1) for i in range(len(file_list))], 'cross individual confusion matrix _%s_%s'%(session, modal))
def test_individual_use_All():
    """
    use svm model trained in all individual  to predict samples of single individual
    :return:
    """
    modal = 'eeg'
    model_load_path = '../../saved_models/unimodal/1'
    file_list = [e for e in os.listdir(model_load_path) if e[-3:] == 'mat']
    data_dicts = data_preparation('1')
    confusion_mat = np.zeros(shape=(1, len(file_list)))
    with codecs.open(os.path.join(model_load_path,  'best_svm_on_all_1_use_eeg.pk'), 'rb') as f:
        svm_clf = pickle.load(f)
    for i, individual in enumerate(file_list):
        data_self =data_dicts[file_list[i]]
        _, _, X_, Y_ = make_train_test(data_self, modal)
        confusion_mat[0][i] = np.round(f1_score(Y_, svm_clf.predict(X_), average='macro'), decimals=3)

    plot_confusion_by_heat_map(confusion_mat * 100, ['all'], ['P%d' % (i+1) for i in range(len(file_list))], 'test_individual_use_all ')

# 4 choose data
def train_on_individual(fold='1', modal='eeg'):
    """
    excute main function
    :param fold: indicate which experiment data to use
    :param modal: indicate the modality
    :return: no return
    """
    # label_dict 每个类别对应的trails的下标
    data_dicts = data_preparation(fold)
    train_index, test_index = make_train_test_index(fold, flag='without_balance')
    # print(data_dicts)
    for key in data_dicts.keys():
        print('dealing with %s\n' % key)
        data = data_dicts[key] # 是一个24个元素的列表
        train_X, train_Y, test_X, test_Y = make_train_test(data,train_index,test_index,modal)
        print(train_X.shape)
        print(train_Y.shape)
        print(test_X.shape)
        print(test_Y.shape)
        if modal in ['eeg','eye']:
            save_path = '%s/%s/%s' % (fold, key, modal)
        else:
            save_path = '%s/%s' % (fold, key)
        build_svm(train_X, train_Y, test_X, test_Y, save_path, model_name="best_svm_use_concat")

def train_on_all_1(fold='1', modal='eeg'):
    """
    train on all subjects, but we sample from every individual by same probability
    :param fold: indicate which experiment data to use
    :param modal: indicate the modality
    :return: no return
    """
    data_dicts = data_preparation(fold)
    individuals = list(data_dicts.keys())
    train_indi = individuals[:10]
    test_indi = individuals[10:]

    # print(data_dicts)
    train_X, train_Y, test_X, test_Y = None, None, None, None
    for i, key in enumerate(train_indi):
        data = data_dicts[key]
        for j, arr in enumerate(data):
            if j == 0:
                data_temp = arr
            else:
                data_temp = np.vstack((data_temp, arr))
        if i == 0:
            train = data_temp
        else:
            train = np.vstack((train, data_temp))
    for i, key in enumerate(test_indi):
        data = data_dicts[key]
        for j, arr in enumerate(data):
            if j == 0:
                data_temp = arr
            else:
                data_temp = np.vstack((data_temp, arr))
        if i == 0:
            test = data_temp
        else:
            test = np.vstack((test, data_temp))

    if modal =="eeg":
        train_X = train[:, :310]
        train_Y = np.array(train[:, -1], dtype=np.int)
        test_X = test[:, :310]
        test_Y = np.array(test[:, -1], dtype=np.int)
    else:
        train_X = train[:, 310:-1]
        train_Y = np.array(train[:, -1], dtype=np.int)
        test_X = test[:, 310:-1]
        test_Y = np.array(test[:, -1], dtype=np.int)
    save_path = '%s/' % (fold)
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    print("working 1...")
    print(train_Y)
    print(test_Y)
    build_svm(train_X, train_Y, test_X, test_Y, save_path, model_name="best_svm_on_all_1_use_%s.pk" % modal)

def train_on_all_2(fold='1', modal='eeg'):
    """
    train on all subjects's samples, can not ensure that every individual appear in both train or test set
    :param fold: indicate which experiment data to use
    :param modal: indicate the modality
    :return: no return
    """
    flag = 'eye'
    data_dicts = data_preparation(fold)
    # print(data_dicts)
    data = None
    for i, key in enumerate(data_dicts.keys()):
        if i == 0:
            data = data_dicts[key]
        else:
            data = np.vstack((data, data_dicts[key]))
    train_X, train_Y, test_X, test_Y = make_train_test(data, modal=modal)
    save_path = '%s/' % (fold)
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    print("working 2...")
    build_svm(train_X, train_Y, test_X, test_Y, save_path, model_name="best_svm_on_all_2_use_%s.pk" % modal)


def main():
    #build_svm()
    #test_B_use_A(fold='3', modal='concat')
    train_on_individual(fold='1', modal='concat')
    #train_on_all_1(fold='1', modal='eeg')
    #train_on_all_2(fold='1', modal='eye')
    #test_individual_use_All()
    # a, b = make_train_test_index('1')
    # print(a,b)
if __name__ == '__main__':
    main()






