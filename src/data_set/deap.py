"""
author: ouyangtianxiong
date: 2020/2/27
des: implement a datasets interface for DEAP dataset
"""
__author__ = "ouyangtianxiong"
import sys
sys.path.append('../')
import numpy as np
from Common_utils.basic_utils import fill_ndarray
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io as scio
import codecs
import pickle


class DEAP:
    """"
    this class purely implement data reading for single individual in single session
    more complex logic should implemented by afterward program
    """
    def __init__(self, individual=1,  normalization=1):
        """
        initial function prepared data we need according to the condition
        :param session: session in [1, 2, 3] means which session of experiment
        :param individual: there are 15 individuals participate this experiment,
                            this parameter reply number 1 to 15
        :param modal: in ['eeg', 'eye']
        :param balance: weather balance the emotion distribution for training set and testing set
        """
        assert individual in [i for i in range(1, 33)], "wrong individual parameter, please check!"
        self.individual = individual
        self.nor_method = normalization

        print("*" * 50)

        # file path of fusion_feature_dict, this data is a type of dict
        # in which every key-value maps to a list contain 24 trials for a experiment individual
        #
        base_path = "../../multimodal_data/DEAP/data_preprocessed_python/s%02d.dat" % (individual)
        with codecs.open(base_path,'rb') as f:
            self.data_dict = pickle.load(f, encoding='ISO-8859-1')# shape:(40,40,8064):(movies num, channel, 63 second * 128HZ)
        self.original_data = self.data_dict['data']
        self.original_label = self.data_dict['labels']
        print(self.original_data.shape)
        base_singal = self.original_data[:, :, :384].mean(axis=-1) # 前三秒的采样
        DATA = self.original_data[:, :, 384:] - base_singal[:, :, np.newaxis]
        print(DATA.shape)
        self.feature = self.normalization(DATA, self.nor_method)
        self.feature = self.feature.reshape(40, 40, 60, -1)
        self.original_label = (self.original_label > 5) + 0
        new_label = np.zeros(shape=(40,1))
        for i in range(40):
            if self.original_label[i][0] == 0 and self.original_label[i][1] == 0:
                new_label[i][0] = 0# 0 for low valence low arousal LALV
            elif self.original_label[i][0] == 0 and self.original_label[i][1] == 1:
                new_label[i][0] = 1# 1 for low valence high arousal HALV
            elif self.original_label[i][0] == 1 and self.original_label[i][1] == 0:
                new_label[i][0] = 2# 2 for high valence low arousal LAHV
            else:
                new_label[i][0] = 3# 3 for high valence hign arousal HAHV
        self.label = np.hstack((self.original_label, new_label))
        self.label = np.expand_dims(self.label, 1).repeat(repeats=60,axis=1)
        self.print_des()

    def normalization(self, data, flag=0):
        """
        apply normalization operator on data
        :param data: transformation data
        :param flag: normalization method 0 for maxmin 1 for 0 means 1 std
        :return:
        """
        for i in range(40):
            for j in range(40):
                temp_col = data[i][j]  # 取出当前列
                nan_num = np.count_nonzero(temp_col != temp_col)  # 判断当前列中是否含nan值
                if nan_num != 0:
                    temp_not_nan_col = temp_col[temp_col == temp_col]
                    temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()  # 用其余元素的均值填充nan所在位置
        if flag == 0:
            np.mean(data, axis=-1, keepdims=True)
            data = (data - np.min(data, axis=-1, keepdims=True)) / (np.max(data, axis=-1, keepdims=True) - np.min(data, axis=-1, keepdims=True) + 1e-8)
        elif flag == 1:
            data = (data - np.mean(data, axis=-1, keepdims=True)) / (np.std(data, axis=-1, keepdims=True) + 1e-8)
        else:
            return data
        return data

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass

    def get_X_Y(self):
        return self.feature.astype(np.float32), self.label

    def print_des(self):
        print("dataset parameters : \nindividual\t%d\nnormalization:\t%s" %
              ( self.individual, self.nor_method))

class DEAP_SEG(DEAP):
    def __init__(self, individual=1, normalization=1):
        pass

    def segmentation(self,flag='train'):
        pass


    #@override
    def get_train_data(self):
        pass

    #@override
    def get_test_data(self):
        pass

    #@override
    def get_X_Y(self):
        pass

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
    individual = 4
    shuffle = True
    nor_method = 0
    deap_data = DEAP(1,nor_method)
    X, Y = deap_data.get_X_Y()
    for i in range(40):
        base_index = list(range(40))
        test_index = base_index[i]
        base_index.remove(i)
        train_X, train_Y = X[base_index], Y[base_index]
        test_X, test_Y = X[test_index], Y[test_index]
        train_X, train_Y = train_X.reshape(-1, 40, 128), train_Y.reshape(-1,5)
        test_X, test_Y = test_X.reshape(-1, 40, 128), test_Y.reshape(-1,5)
        print("%d 作为测试集"%i, train_X.shape, test_X.shape, train_Y.shape,test_Y.shape)
if __name__ == '__main__':
    main()
