import sys
sys.path.append("..")
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from data_set.deap_feature import DEAP, DEAP_DATASET
from data_set.seed_iv import SEED_IV
import pandas as pd
import os
from Common_utils.basic_utils import seed_normalization, deap_normalization
import numpy as np



def build_classifier():
    c_range = np.logspace(-10, 10, 11, base=2)
    gamma_range = np.logspace(-10, 2, 11, base=2)
    kernel = ['rbf']
    parameters = {'C': c_range, "kernel":kernel}
    print("building svm classifier")
    svm_clf = SVC()
    gs = GridSearchCV(svm_clf, param_grid=parameters, refit=True, n_jobs=-1,  verbose=2)
    return gs

def train_on_single_subject_deap(individual=1, class_target=0, k_fold=10):
    class_list = [
        "Valence",
        "Arousal",
        "Dominance",
        "Liking",
        "Valence-Arousal"
    ]

    if not os.path.exists('./deap2_cv_results/individual%02d'%individual):
        os.makedirs('./deap2_cv_results/individual%02d' % individual)

    svm_clf = build_classifier()
    test_acc_list = []
    test_precision_list = []
    test_recall_list = []
    test_f1_list = []
    deap = DEAP(individual=individual)
    k_fold_data = deap.get_kfold_X_Y(k_fold)
    for fold, (train_X, train_Y, test_X, test_Y) in enumerate(k_fold_data):
        print("start {} th cross-validation".format(fold))
        train_X, train_Y, test_X, test_Y = deap_normalization(train_X, train_Y, test_X, test_Y, nor_method=0, merge=1,
                                                              column=0)
        train_Y, test_Y = train_Y[:, class_target], test_Y[:, class_target]
        print("train Y unique{}\t test Y unique{}".format(np.unique(train_Y), np.unique(test_Y)))
        svm_clf.fit(train_X, train_Y)
        predict = svm_clf.predict(test_X)
        test_acc_list.append(accuracy_score(test_Y, predict))
        test_precision_list.append(precision_score(test_Y, predict, average='macro'))
        test_recall_list.append(recall_score(test_Y, predict, average="macro"))
        test_f1_list.append(f1_score(test_Y, predict,average="macro"))
        print("classification report\n{}\nACC {}\nconfusion matrix\n{}\n".format(
            classification_report(test_Y, predict, digits=4), accuracy_score(test_Y, predict),
            confusion_matrix(test_Y, predict)))
    df = pd.DataFrame().from_dict({"accuracy":test_acc_list,
                              "precision":test_precision_list,
                              "recall":test_recall_list,
                              "f1":test_f1_list})
    df_mean = df.mean()
    df_std = df.std()
    df = df.append(df_mean, ignore_index=True)
    df = df.append(df_std, ignore_index=True)
    df.to_csv('./deap2_cv_results/individual%02d/%s.csv' % (individual, class_list[class_target]))

def train_on_single_subject_seed(session=1):

    nor_method = 1
    if not os.path.exists('./seed_results/session%02d' % (session)):
        os.makedirs('./seed_results/session%02d' % (session))

    svm_clf = build_classifier()
    test_acc_list = []
    test_precision_list = []
    test_recall_list = []
    test_f1_list = []
    for individual in range(1, 16):
        seed_iv = SEED_IV(individual=individual, session=session, modal="concat",  normalization=nor_method, balance=True, shuffle=False, k_fold=6)
        k_fold_data = seed_iv.get_training_kfold_data()
        for i, (train_X, train_Y, test_X, test_Y) in enumerate(k_fold_data):
            # train_X, train_Y, test_X, test_Y = seed_iv.get_training_data()
            # X, Y = seed_iv.get_X_Y()
            # train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state=2020, test_size=0.2)
            train_X, train_Y, test_X, test_Y = seed_normalization(train_X, train_Y, test_X, test_Y, nor_method=0, merge=1, column=0)

            svm_clf.fit(train_X, train_Y)
            predict = svm_clf.predict(test_X)
            test_acc_list.append(accuracy_score(test_Y, predict))
            test_precision_list.append(precision_score(test_Y, predict, average='macro'))
            test_recall_list.append(recall_score(test_Y, predict, average="macro"))
            test_f1_list.append(f1_score(test_Y, predict, average="macro"))
            print("classification report\n{}\nACC {}\nconfusion matrix\n{}\n".format(classification_report(test_Y, predict, digits=4),accuracy_score(test_Y,predict),
                                                                             confusion_matrix(test_Y, predict)))

    df = pd.DataFrame().from_dict({"accuracy":test_acc_list,
                              "precision":test_precision_list,
                              "recall":test_recall_list,
                              "f1":test_f1_list})
    df_mean = df.mean()
    df_std = df.std()
    df = df.append(df_mean, ignore_index=True)
    df = df.append(df_std, ignore_index=True)
    df.to_csv('./seed_results/session%02d/results.csv' % (session))
if __name__ == "__main__":
    # for session in [1, 2, 3]:
    #     train_on_single_subject_seed(session=session)
    for c in [4, 0, 1, 2]:
        for i in range(1, 33):
            train_on_single_subject_deap(individual=i, class_target=c, k_fold=10)