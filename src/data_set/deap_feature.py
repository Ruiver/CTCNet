"""
author: ouyangtianxiong
date: 2020/3/20
des: implement a datasets interface for DEAP dataset
"""
__author__ = "ouyangtianxiong"
import sys
sys.path.append('../')
from Common_utils.basic_utils import fill_ndarray

#from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
import codecs

def discretization_label(label):
    assert len(label.shape) == 2, "label need to be 2-dimension"
    label = (label > 5) + 0
    new_label = np.zeros(shape=(label.shape[0], 1))
    for i in range(label.shape[0]):
        if label[i][0] == 0 and label[i][1] == 0:
            new_label[i][0] = 0  # 0 for low valence low arousal LALV
        elif label[i][0] == 0 and label[i][1] == 1:
            new_label[i][0] = 1  # 1 for low valence high arousal HALV
        elif label[i][0] == 1 and label[i][1] == 0:
            new_label[i][0] = 2  # 2 for high valence low arousal LAHV
        else:
            new_label[i][0] = 3  # 3 for high valence hign arousal HAHV
    label = np.hstack((label, new_label))
    return label

def deap_processor(individual=1):
    """
    initial function prepared data we need according to the condition
    :param session: session in [1, 2, 3] means which session of experiment
    :param individual: there are 15 individuals participate this experiment,
                        this parameter reply number 1 to 15
    :param modal: in ['eeg', 'eye']
    :param balance: weather balance the emotion distribution for training set and testing set
    """
    assert individual in [i for i in range(1,33)], "wrong individual parameter, please check!"
    individual = individual
    data_training = []
    label_training = []
    data_validation = []
    label_validation = []
    data_testing = []
    label_testing = []

    print("*" * 50)

    # file path of fusion_feature_dict, this data is a type of dict
    # in which every key-value maps to a list contain 24 trials for a experiment individual
    #
    #base_path = "../../multimodal_data/DEAP/out/s%02d.dat" % (individual)
    with open('../../multimodal_data/DEAP/out3/s%02d.npy' % individual, 'rb') as file:
        sub = np.load(file, allow_pickle=True)# shape:(40 * samples, 200)
        #sub = sub.reshape(40, -1, 200) # (video, samples, 200)
        # np.random.shuffle(sub)
        for i in range(0, sub.shape[0]):
            if i % 8 == 0:
                data_testing.append(sub[i][0])
                label_testing.append(sub[i][1])
            elif i % 8 == 1:
                data_validation.append(sub[i][0])
                label_validation.append(sub[i][1])
            else:
                data_training.append(sub[i][0])
                label_training.append(sub[i][1])
    data_training = np.array(data_training)
    data_validation = np.array(data_validation)
    data_testing = np.array(data_testing)
    label_training = np.array(label_training)
    label_validation = np.array(label_validation)
    label_testing = np.array(label_testing)

    print(data_training.shape, label_training.shape)
    print(data_validation.shape, label_validation.shape)
    print(data_testing.shape, label_testing.shape)

    label_training = discretization_label(label_training)
    label_validation = discretization_label(label_validation)
    label_testing = discretization_label(label_testing)

    print(data_training.shape, label_training.shape)
    print(data_validation.shape, label_validation.shape)
    print(data_testing.shape, label_testing.shape)

    if not os.path.exists('../../multimodal_data/DEAP/out3/s%02d'%individual):
        os.makedirs('../../multimodal_data/DEAP/out3/s%02d/'%individual)

    np.save('../../multimodal_data/DEAP/out3/s%02d/data_training'%individual, np.array(data_training), allow_pickle=True, fix_imports=True)
    np.save('../../multimodal_data/DEAP/out3/s%02d/label_training'%individual, np.array(label_training), allow_pickle=True, fix_imports=True)

    np.save('../../multimodal_data/DEAP/out3/s%02d/data_validation'%individual, np.array(data_validation), allow_pickle=True, fix_imports=True)
    np.save('../../multimodal_data/DEAP/out3/s%02d/label_validation'%individual, np.array(label_validation), allow_pickle=True, fix_imports=True)

    np.save('../../multimodal_data/DEAP/out3/s%02d/data_testing'%individual, np.array(data_testing), allow_pickle=True, fix_imports=True)
    np.save('../../multimodal_data/DEAP/out3/s%02d/label_testing'%individual, np.array(label_testing), allow_pickle=True, fix_imports=True)

class DEAP:
    """"
    this class purely implement data reading for single individual in single session
    more complex logic should implemented by afterward program
    """
    def __init__(self, individual=1):
        """
        initial function prepared data we need according to the condition
        :param individual: there are 15 individuals participate this experiment,
                            this parameter reply number 1 to 15
        """
        assert individual in [i for i in range(1,33)], "wrong individual parameter, please check!"
        self.individual = individual
        print("*" * 50)

        # file path of fusion_feature_dict, this data is a type of dict
        # in which every key-value maps to a list contain 24 trials for a experiment individual
        #
        base_dir = "../../multimodal_data/DEAP/out_persec_reshape/s%02d" % (self.individual)
        with open(base_dir + '.npy', 'rb') as file:
            self.data_list = np.load(file, allow_pickle=True)
        self.print_des()

    def get_train_data(self):
        pass
    def get_validate_data(self):
        pass
    def get_test_data(self):
        pass
    def get_kfold_X_Y(self, k_fold, target_label=0):
        def zero_mean(data):
            feature = data[:, :-1]
            label = data[:, -1]
            print("feature shape {}\tlabel shape {}".format(feature.shape, label.shape))
            mean = np.mean(feature, axis=0)
            std = np.std(feature, axis=0)
            feature = (feature - mean)
            data = np.hstack((feature, label.reshape(-1, 1)))
            print("feature shape {}\tlabel shape {}\tdata shape {}".format(feature.shape, label.shape, data.shape))
            return data

        all_label = [e[0][1] for e in self.data_list]
        all_label = np.vstack(all_label)
        all_label = discretization_label(all_label)
        label_distribute = {}
        for i in range(len(all_label)):
            if all_label[i][target_label] in label_distribute:
                label_distribute[all_label[i][target_label]].append(i)
            else:
                label_distribute[all_label[i][target_label]] = [i]
        k_fold_data = []
        for i in range(k_fold):
            train_X = []
            train_Y = []
            test_X = []
            test_Y = []
            for j in range(len(self.data_list)):

                if j % k_fold == i:
                    for k in range(self.data_list[j].shape[0]):
                        test_X.append(self.data_list[j][k][0])
                        test_Y.append(self.data_list[j][k][1])
                else:
                    for k in range(self.data_list[j].shape[0]):
                        train_X.append(self.data_list[j][k][0])
                        train_Y.append(self.data_list[j][k][1])
            train_X = np.array(train_X).astype(np.float32)
            test_X = np.array(test_X).astype(np.float32)
            train_Y = np.array(train_Y).astype(np.int32)
            test_Y = np.array(test_Y).astype(np.int32)
            train_Y = discretization_label(train_Y)
            test_Y = discretization_label(test_Y)
            k_fold_data.append([train_X, train_Y, test_X, test_Y])
        return k_fold_data

    def get_kfold_X_Y2(self, k_fold):
        k_fold_data = []
        X, Y = [], []
        for j in range(len(self.data_list)):
            for k in range(self.data_list[j].shape[0]):
                X.append(self.data_list[j][k][0])
                Y.append(self.data_list[j][k][1])
        X = np.array(X)
        Y = np.array(Y)
        Y = discretization_label(Y)
        X = np.array(X).astype(np.float32)
        Y = np.array(Y).astype(np.int32)
        for i in range(k_fold):
            train_X, train_Y, test_X, test_Y = [], [], [], []
            for j in range(X.shape[0]):
                if j % k_fold == i:
                    test_X.append(X[j])
                    test_Y.append(Y[j])
                else:
                    train_X.append(X[j])
                    train_Y.append(Y[j])
            train_X = np.array(train_X).astype(np.float32)
            train_Y = np.array(train_Y).astype(np.int32)
            test_X = np.array(test_X).astype(np.float32)
            test_Y = np.array(test_Y).astype(np.int32)
            k_fold_data.append([train_X, train_Y, test_X, test_Y])
        return k_fold_data
    def print_des(self):
        print("dataset parameters : \nindividual\t%d" %
              ( self.individual))

class DEAP128:
    """"
    this class purely implement data reading for single individual in single session
    more complex logic should implemented by afterward program
    """
    def __init__(self, individual=1):
        """
        initial function prepared data we need according to the condition
        :param individual: there are 15 individuals participate this experiment,
                            this parameter reply number 1 to 15
        """
        assert individual in [i for i in range(1,33)], "wrong individual parameter, please check!"
        self.individual = individual
        print("*" * 50)

        # file path of fusion_feature_dict, this data is a type of dict
        # in which every key-value maps to a list contain 24 trials for a experiment individual
        #
        base_dir = "../../multimodal_data/DEAP/data_preprocessed_python/s%02d" % (self.individual)
        # with open(base_dir + '.npy', 'rb') as file:
        #     self.data_list = np.load(file, allow_pickle=True)
        with codecs.open('../../multimodal_data/DEAP/data_preprocessed_python/s01.dat', 'rb') as f:
            x = pickle.load(f, encoding="ISO-8859-1")
        print(type(x))
        #self.feature = x['data'].reshape(40, 40, -1, 128).transpose(0, 2, 1, 3).reshape(40,63,-1)
        # (video, channel, second, 128) -> (video, second, channel, 128)
        self.feature = x['data'].reshape(40, 40, -1, 63).transpose(0, 3, 1, 2).reshape(40, 63, -1)
        #(video, channel,  128, second) -> (video, second, channel, 128)
        self.label = x['labels']
        print(x['labels'].shape)
        print(x['data'].shape)
        self.print_des()

    def get_train_data(self):
        pass
    def get_validate_data(self):
        pass
    def get_test_data(self):
        pass
    def get_kfold_X_Y(self, k_fold, target_label=0):
        def zero_mean(data):
            feature = data[:, :-1]
            label = data[:, -1]
            print("feature shape {}\tlabel shape {}".format(feature.shape, label.shape))
            mean = np.mean(feature, axis=0)
            std = np.std(feature, axis=0)
            feature = (feature - mean)
            data = np.hstack((feature, label.reshape(-1, 1)))
            print("feature shape {}\tlabel shape {}\tdata shape {}".format(feature.shape, label.shape, data.shape))
            return data
        k_fold_data = []
        video_idx = list(range(40))
        np.random.shuffle(video_idx)
        for i in range(k_fold):
            train_X = []
            train_Y = []
            test_X = []
            test_Y = []
            for j in range(len(self.feature)):
                if j % k_fold == i:
                    for k in range(self.feature[video_idx[j]].shape[0]):
                        test_X.append(self.feature[video_idx[j]][k])
                        test_Y.append(self.label[video_idx[j]])
                else:
                    for k in range(self.feature[video_idx[j]].shape[0]):
                        train_X.append(self.feature[video_idx[j]][k])
                        train_Y.append(self.label[video_idx[j]])
            train_X = np.array(train_X).astype(np.float32)
            test_X = np.array(test_X).astype(np.float32)
            train_Y = np.array(train_Y).astype(np.int32)
            test_Y = np.array(test_Y).astype(np.int32)
            train_Y = discretization_label(train_Y)
            test_Y = discretization_label(test_Y)
            k_fold_data.append([train_X, train_Y, test_X, test_Y])
        return k_fold_data

    def get_kfold_X_Y2(self, k_fold):
        k_fold_data = []
        X, Y = [], []
        for j in range(len(self.feature)):
            for k in range(self.feature[j].shape[0]):
                X.append(self.feature[j][k])
                Y.append(self.label[j])
        X = np.array(X)
        Y = np.array(Y)
        Y = discretization_label(Y)
        X = np.array(X).astype(np.float32)
        Y = np.array(Y).astype(np.int32)
        idx = list(range(len(X)))
        np.random.shuffle(idx)
        for i in range(k_fold):
            train_X, train_Y, test_X, test_Y = [], [], [], []
            for j in range(X.shape[0]):
                if j % k_fold == i:
                    test_X.append(X[idx[j]])
                    test_Y.append(Y[idx[j]])
                else:
                    train_X.append(X[idx[j]])
                    train_Y.append(Y[idx[j]])
            train_X = np.array(train_X).astype(np.float32)
            train_Y = np.array(train_Y).astype(np.int32)
            test_X = np.array(test_X).astype(np.float32)
            test_Y = np.array(test_Y).astype(np.int32)
            k_fold_data.append([train_X, train_Y, test_X, test_Y])
        return k_fold_data
    def print_des(self):
        print("dataset parameters : \nindividual\t%d" %
              ( self.individual))


class DEAP_DATASET(Dataset):
    def __init__(self, X, Y):
        """
        transform the original data to datasets from
        :param X: train_X
        :param Y: train_Y
        """
        assert len(X) == len(Y), "n_samples dimension mismatch"
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

def main():
    individual = 11
    nor_method = 1
    cls = 4
    deap_data = DEAP128(individual)
    k_fold_data = deap_data.get_kfold_X_Y(k_fold=10)
    print("k_fold_data shape {}".format(len(k_fold_data)))
    for i,(train_X, train_Y, test_X, test_Y) in enumerate(k_fold_data):
        print("train X shape {}".format(train_X.shape))
        print("train Y shape {}".format(train_Y.shape))
        print("test X shape {}".format(test_X.shape))
        print("test Y shape {}".format(test_Y.shape))
        print("train Y unique {}\t test Y unique {}".format(np.unique(train_Y), np.unique(test_Y)))
    # #print(X[:10])
    # #print(np.mean(X,axis=0), np.std(X,axis=0), np.min(X,axis=0),np.max(X,axis=0))
    # Y = Y[:, cls]
    # a, b, c, d  = sum(Y == 0), sum(Y == 1), sum(Y == 2), sum(Y == 3)
    # print('训练集batch, 类别为0的有：{},占比约为:{}'.format(a, a / (a+b+c+d)))
    # print('训练集batch, 类别为1的有：{},占比约为:{}'.format(b, b / (a+b+c+d)))
    # print('训练集batch, 类别为2的有：{},占比约为:{}'.format(c, c / (a+b+c+d)))
    # print('训练集batch, 类别为3的有：{},占比约为:{}'.format(d, d / (a+b+c+d)))
    # train_loader = DataLoader(dataset=DEAP_DATASET(X, Y), shuffle=True, batch_size=512)
    # X, Y = deap_data.get_validate_data()
    # Y = Y[:, cls ]
    # a, b, c, d = sum(Y == 0), sum(Y == 1), sum(Y == 2), sum(Y == 3)
    # print('验证集batch, 类别为0的有：{},占比约为:{}'.format(a, a / (a + b + c + d)))
    # print('验证集batch, 类别为1的有：{},占比约为:{}'.format(b, b / (a + b + c + d)))
    # print('验证集batch, 类别为2的有：{},占比约为:{}'.format(c, c / (a + b + c + d)))
    # print('验证集batch, 类别为3的有：{},占比约为:{}'.format(d, d / (a + b + c + d)))
    # validate_loader = DataLoader(dataset=DEAP_DATASET(X, Y),shuffle=True, batch_size=512)
    # X, Y = deap_data.get_test_data()
    # Y = Y[:, cls]
    # a, b, c, d = sum(Y == 0), sum(Y == 1), sum(Y == 2), sum(Y == 3)
    # print('测试集batch, 类别为0的有：{},占比约为:{}'.format(a, a / (a + b + c + d)))
    # print('测试集batch, 类别为1的有：{},占比约为:{}'.format(b, b / (a + b + c + d)))
    # print('测试集batch, 类别为2的有：{},占比约为:{}'.format(c, c / (a + b + c + d)))
    # print('测试集batch, 类别为3的有：{},占比约为:{}'.format(d, d / (a + b + c + d)))
    # test_loader = DataLoader(dataset=DEAP_DATASET(X, Y), shuffle=True, batch_size=512)
    # for i, (feature, target) in enumerate(train_loader):
    #     x = feature[0]
    #     x = x.reshape(40, 5)
    #     #print("sample: \n{}".format(x))
    #     uniqu_array = np.unique(target.numpy())
    #     target = target.numpy()
    #     print('训练集loader，batch：{}, 样本数:{}'.format(i, len(target)))
    #     print('训练集的标签集为{} 不重复的个数有：{}'.format(uniqu_array, len(uniqu_array)))
    #     for j in uniqu_array:
    #         print(np.sum(target == j))
    # v_a = [0, 0, 0, 0]
    # for i, (feature, target) in enumerate(validate_loader):
    #     uniqu_array = np.unique(target.numpy())
    #     target = target.numpy()
    #     x = feature[0]
    #     x = x.reshape(40, 5)
    #     #print("sample: \n{}".format(x))
    #     print('验证集loader，batch：{}, 样本数:{}'.format(i, len(target)))
    #     print('训练集的标签集为{} 不重复的个数有：{}'.format(uniqu_array, len(uniqu_array)))
    #     for j in uniqu_array:
    #         v_a[int(j)] += np.sum(target==j)
    #         print(np.sum(target==j))
    # print(v_a)
    # t_a = [0, 0, 0, 0]
    # for i, (feature, target) in enumerate(test_loader):
    #     uniqu_array = np.unique(target.numpy())
    #     target = target.numpy()
    #     x = feature[0]
    #     x = x.reshape(40, 5)
    #     #print("sample: \n{}".format(x))
    #     print('测试集loader，batch：{}, 样本数:{}'.format(i, len(target)))
    #     print('训练集的标签集为{} 不重复的个数有：{}'.format(uniqu_array, len(uniqu_array)))
    #     for j in uniqu_array:
    #         t_a[int(j)] += np.sum(target==j)
    #         print(np.sum(target==j))
    # print(t_a)
if __name__ == '__main__':

    main()
    # for i in range(1, 33):
    #     deap_processor(i)
    # for (train_data, test_data) in DEAP().get_kfold_X_Y(k_fold=5):
    #     assert len(train_data) == len(test_data) == 2, "wrong shape"
    #     print("train data shape {}, train label shape {}\n".format(train_data[0].shape, train_data[1].shape))
    #     print("test data shape {}, test label shape {}\n".format(test_data[0].shape, test_data[1].shape))

