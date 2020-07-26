"""
author: ouyangtianxiong
date: 2019/12/17
des: implement a datasets interface for SEED dataset
"""
__author__ = "ouyangtianxiong"
import sys
sys.path.append('../')
import numpy as np
import random
from Common_utils.basic_utils import fill_ndarray
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io as scio

class SEED:
    """"
    this class purely implement data reading for single individual in single session
    more complex logic should implemented by afterward program
    """
    def __init__(self,session=1, individual=1, balance=True,shuffle=False,normalization=1):
        """
        initial function prepared data we need according to the condition
        :param session: session in [1, 2, 3] means which session of experiment
        :param individual: there are 15 individuals participate this experiment,
                            this parameter reply number 1 to 15
        :param modal: in ['eeg', 'eye']
        :param balance: weather balance the emotion distribution for training set and testing set
        """
        assert session in [1,2,3], "wrong session parameter, please check!"
        assert individual in [i for i in range(1,16)], "wrong individual parameter, please check!"
        assert isinstance(balance, bool), "wrong balance parameter, please check "

        self.session = session
        self.individual = individual
        self.balance = balance
        self.shuffle = shuffle
        self.nor_method = normalization

        # 这几个列表用来存储未混合起来的trails级的参数每个元素是一个trail的数据
        self.trails_train = []
        self.trails_test = []

        print("*" * 50)
        self.print_des()
        # file path of fusion_feature_dict, this data is a type of dict
        # in which every key-value maps to a list contain 24 trials for a experiment individual
        #
        base_path = "../../multimodal_data/SEED/ExtractedFeatures"
        files = os.listdir(base_path)
        indis_files = [e for e in files if e.split('_')[0] == str(individual)]
        indis_files = sorted(indis_files)
        file_path = indis_files[session-1]
        data_dict = scio.loadmat(os.path.join(base_path,file_path),verify_compressed_data_integrity =False)
        assert isinstance(data_dict, dict), "data_dict is not the type of dict"
        data_list = []
        for i in range(1,16):
            key = "de_LDS%d"%i
            data = data_dict[key]
            b = data.shape[1]
            data = data.transpose((1, 0, 2)).reshape(b, -1)
            data_list.append(data)
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.make_train_test(data_list,self.balance,shuffle=self.shuffle)
        print("data construct successful...")
        print("*" * 50)
    def make_train_test(self,data, flag=True, shuffle=False):
        labels_distribute = [1,  0, -1, -1,  0,  1, -1,  0,  1,  1,  0, -1,  0,  1, -1]
        data = [np.insert(a, 310, labels_distribute[i]+1, axis=1) for i,a in enumerate(data)]
        if flag:
            current_label = labels_distribute
            label_dict = {k: [] for k in set(current_label)}
            for i, l in enumerate(current_label):
                label_dict[l].append(i)
            if shuffle is True:
                for _,v in label_dict.items():
                    random.shuffle(v)
            train_index = []
            test_index = []
            for k, v in label_dict.items():
                train_index.extend(v[:3])
                test_index.extend(v[3:])
        else:
            idx = list(range(len(labels_distribute)))
            train_index = idx[:9]
            test_index = idx[9:]
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]
        self.trails_train = train_data
        self.trails_test = test_data
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
        train_X = train[:, :-1]
        train_Y = train[:, -1]
        test_X = test[:, :-1]
        test_Y = test[:, -1]
        train_Y, test_Y = np.array(train_Y, dtype=np.int), np.array(test_Y, dtype=np.int)
        return train_X, train_Y, test_X, test_Y

    def normalization(self,data, flag=0):
        """
        apply normalization operator on data
        :param data: transformation data
        :param flag: normalization method 0 for maxmin 1 for 0 means 1 std
        :return:
        """
        data = fill_ndarray(data)
        if flag == 0:
            data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        elif flag == 1:
            data = (data - data.mean(axis=0)) / (data.std(axis=0))
        else:
            return  data
        return data

    def get_train_data(self):
        return self.normalization(self.train_X.astype(np.float32),flag=self.nor_method), self.train_Y

    def get_test_data(self):
        return self.normalization(self.test_X.astype(np.float32),flag=self.nor_method), self.test_Y

    def get_X_Y(self):
        # print(self.train_X.shape)
        # print(self.train_Y.shape)
        # print(self.test_X.shape)
        # print(self.test_Y.shape)
        X = np.vstack((self.train_X, self.test_X)).astype(np.float32)
        # print(X.shape)
        Y = np.append(self.train_Y, self.test_Y)
        return self.normalization(X,flag=self.nor_method), Y

    def print_des(self):
        print("dataset parameters : \nmodality\t%s\nindividual\t%d\nsession\t%d\nemotion distribution\t%s\nemotion shuffle\t%s\nnormalization:\t%s" %
              ("eeg", self.individual, self.session,"balanced" if self.balance else "unbalanced", "yes" if self.shuffle else "no",self.nor_method))
        print("emotion class info:\n1 for negative emotion\n2 for neutral emotion\n3 for positive emotion\n")
class SEED_SEG(SEED):
    def __init__(self,T_len,session=1, individual=1, balance=True,shuffle=False,normalization=1):
        self.T = T_len
        super(SEED_SEG, self).__init__(session=session,individual=individual,balance=balance,shuffle=shuffle,normalization=normalization)
        trails_train = self.trails_train
        trails_test = self.trails_test

    def segmentation(self,flag='train'):
        # 按照T分段
        # X, Y (B,310) (B,1)
        # 遍历每一个trail
        X = None
        Y = None
        if flag == 'train':
            data = self.trails_train
        elif flag == 'test':
            data = self.trails_test
        elif flag == 'all':
            data = self.trails_train + self.trails_test
        else:
            raise ValueError
        for i, trail in enumerate(data):
            # trail numpy数组 (sample, 310)
            if i == 0:
                for j in range(trail.shape[0] - self.T + 1):
                    if j == 0:
                        X = trail[j:j + self.T, :310].reshape(1, -1)
                        Y = trail[j][-1]
                    else:
                        X = np.vstack((X, trail[j:j + self.T, :310].reshape(1, -1)))
                        Y = np.append(Y, trail[j][-1])
            else:
                for j in range(trail.shape[0] - self.T + 1):
                    X = np.vstack((X, trail[j:j + self.T, :310].reshape(1, -1)))
                    Y = np.append(Y, trail[j][-1])

        print(X.shape, Y.shape)
        return self.normalization(X.astype(np.float32).reshape(-1,self.T, 310), flag=self.nor_method).reshape(-1,self.T * 310), Y.astype(np.int32)
    #@override
    def get_train_data(self):
        return self.segmentation(flag='train')

    #@override
    def get_test_data(self):
        return self.segmentation(flag='test')

    #@override
    def get_X_Y(self):
        return self.segmentation(flag='all')

class SEED_DATASET(Dataset):
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
    session = 1
    balance = False
    shuffle = True
    nor_method = 1
    # seed_iv = SEED(individual=individual, session=session,balance=balance)
    # train_X, train_Y = seed_iv.get_train_data()
    # test_X, test_Y = seed_iv.get_test_data()
    # X, Y = seed_iv.get_X_Y()
    # print()
    # X = (X - X.min()) / (X.max() - X.min())
    # print(X.shape, Y.shape)
    # seed_iv_train_dataset = SEED_DATASET(X=train_X, Y=train_Y)
    # seed_iv_test_dataset = SEED_DATASET(X=test_X, Y=test_Y)
    # seed_iv_train_loader = DataLoader(dataset=seed_iv_train_dataset, shuffle=False,batch_size=32,num_workers=4)
    # seed_iv_test_loader = DataLoader(dataset=seed_iv_test_dataset, shuffle=True, batch_size=32,num_workers=4)
    #
    #
    # for i, (features, target) in enumerate(seed_iv_train_loader):
    #     print("batch %d" % (i+1))
    #     print(features.shape)
    #     print(target.shape)
    #     print(target)
    # for i, (features, target) in enumerate(seed_iv_test_loader):
    #     print("batch %d" % (i+1))
    #     print(features.shape)
    #     print(target.shape)
    #     print(target)
    data = SEED_SEG(T_len=9,individual=individual, session=session,balance=balance,normalization=nor_method)
    X,Y = data.get_train_data()
    print(X.mean(axis=0), X.std(axis=0))
    print(X.reshape(-1,9,310).mean(axis=0), X.reshape(-1,9,310).std(axis=0))

    loader = DataLoader(dataset=SEED_DATASET(X=X, Y=Y),batch_size=32,shuffle=shuffle,num_workers=4)
    print(len(loader), type(len(loader)))

    for i, (features, target) in enumerate(loader):
        print("batch %d" % (i+1))
        print(features.shape)
        print(target.shape)
        print(target)

if __name__ == '__main__':
    main()




