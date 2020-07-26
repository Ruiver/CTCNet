"""
author: ouyangtianxiong
date: 2019/12/07
des: implement a datasets interface for SEED_IV dataset
"""
import numpy as np
import random
import pickle
import codecs
import sys
sys.path.append('../')

from torch.utils.data import Dataset, DataLoader, random_split

class SEED_IV:
    """"
    this class purely implement data reading for single individual in single session
    more complex logic should implemented by afterward program
    """
    def __init__(self, session=1, individual=1, modal='concat', balance=False, shuffle=False, normalization=1, k_fold=0):
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
        assert modal in ['eeg', 'eye', 'concat'], "wrong modal parameter, please check!"
        assert isinstance(balance, bool), "wrong balance parameter, please check "
        self.session = session
        self.individual = individual
        self.modal = modal
        self.balance = balance
        self.shuffle = shuffle
        self.nor_method = normalization
        self.k_fold = k_fold

        # 这几个列表用来存储未混合起来的trails级的参数每个元素是一个trail的数据

        print("*" * 50)
        self.print_des()
        # file path of fusion_feature_dict, this data is a type of dict
        # in which every key-value maps to a list contain 24 trials for a experiment individual
        #
        base_path = "../../multimodal_feature_extracted/SEED_IV/%d/fusion_feature.pk" % session
        with codecs.open(base_path, mode='rb') as f:
            data_dict = pickle.load(f)
        assert isinstance(data_dict, dict), "data_dict is not the type of dict"
        keys = data_dict.keys()
        indis = {key.split('_')[0]: key.split('_')[1] for key in keys}
        individual_name = "%s_%s" % (individual, indis[str(individual)])
        self.data_list = data_dict[individual_name]
        if k_fold == 0:
            self.train_X, self.train_Y, self.test_X, self.test_Y = self.make_train_test(self.data_list, self.session,self.modal,self.balance,shuffle=self.shuffle)
        else:
            self.k_fold_data = self.get_Kfold_data(self.data_list)
        print("data construct successful...")
        print("*" * 50)

    def get_Kfold_data(self, data_list):
        labels_distribute = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                             [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                             [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
        current_label = labels_distribute[int(self.session) - 1]
        label_dict = {k: [] for k in set(current_label)}
        for i, l in enumerate(current_label):
            label_dict[l].append(i)
        if self.shuffle:
            for _, v in label_dict.items():
                random.shuffle(v) #序号做shuffle
        k_fold_Data = []
        for fold in range(self.k_fold):
        # 做六折验证， 每一折选取每类情感下的第fold个video作为测试集
            train_data = []
            test_data = []
            for k, v in label_dict.items():
                # 遍历每类情感下的video序号
                for i in range(6):
                    if i % self.k_fold == fold:
                        test_data.append(data_list[v[i]])
                    else:
                        train_data.append(data_list[v[i]])
            train_data = np.vstack(train_data)
            test_data = np.vstack(test_data)
            if self.modal == "eeg":
                train_X = train_data[:, :310]
                test_X = test_data[:, :310]
            elif self.modal == "eye":
                train_X = train_data[:310:-1]
                test_X = test_data[:, 310:-1]
            else:
                train_X = train_data[:, :-1]
                test_X = test_data[:, :-1]
            train_Y = train_data[:, -1]
            test_Y = test_data[:, -1]
            k_fold_Data.append([train_X, train_Y, test_X, test_Y])
        return k_fold_Data

    def make_train_test(self, data, session, modal='eeg', flag=True, shuffle=False):
        labels_distribute = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                             [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                             [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
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
        if flag:
            current_label = labels_distribute[int(session) - 1]
            label_dict = {k: [] for k in set(current_label)}
            for i, l in enumerate(current_label):
                label_dict[l].append(i)
            if shuffle is True:
                for _, v in label_dict.items():
                    random.shuffle(v)
            train_index = []
            test_index = []
            for k, v in label_dict.items():
                train_index.extend(v[:5])
                test_index.extend(v[5:])
        else:
            idx = list(range(len(labels_distribute[int(session) - 1])))
            train_index = idx[:20]
            test_index = idx[20:]
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]
        # train_data = [zero_mean(data[i]) for i in train_index]
        # test_data = [zero_mean(data[i]) for i in test_index]
        self.trails_train = train_data
        self.trails_test = test_data
        train = np.vstack(train_data)
        test = np.vstack(test_data)
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
        else:
            train_X = train[:, :-1]
            train_Y = train[:, -1]
            test_X = test[:, :-1]
            test_Y = test[:, -1]
        train_Y, test_Y = np.array(train_Y, dtype=np.int), np.array(test_Y, dtype=np.int)
        return train_X, train_Y, test_X, test_Y

    def get_training_data(self):
        return self.train_X.astype(np.float32), self.train_Y.astype(np.int32), self.test_X.astype(np.float32), self.test_Y.astype(np.int32)

    def get_training_kfold_data(self):
        return [[e[0].astype(np.float32), e[1].astype(np.int32),e[2].astype(np.float32), e[3].astype(np.int32)] for e in self.k_fold_data]
    def get_X_Y(self):
        # print(self.train_X.shape)
        # print(self.train_Y.shape)
        # print(self.test_X.shape)
        # print(self.test_Y.shape)
        data = np.vstack(self.data_list)
        if self.modal == "eeg":
            X = data[:, :310]
        elif self.modal == "eye":
            X = data[:.310:-1]
        else:
            X = data[:,:-1]
        Y = data[:, -1]
        X = X.astype(np.float32)
        # print(X.shape)
        Y = Y.astype(np.int32)
        return X, Y

    def print_des(self):
        print("dataset parameters : \nmodality\t%s\nindividual\t%d\nsession\t%d\nemotion distribution\t%s\nemotion shuffle\t%s\nnormalization:\t%s" %
              (self.modal, self.individual, self.session,"balanced" if self.balance else "unbalanced", "yes" if self.shuffle else "no",self.nor_method))
        print("emotion class info:\n0 for neutral\n1 for sad\n2 for fear\n3 for hapy")
class SEED_IV_SEG(SEED_IV):
    def __init__(self,T_len,session=1, individual=1, modal='eeg', balance=True, shuffle=False, normalization=1):
        self.T = T_len
        super(SEED_IV_SEG, self).__init__(session=session,individual=individual,modal=modal,balance=balance,shuffle=shuffle,normalization=normalization)
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
                        X = trail[j:j + self.T, :-1].reshape(1, -1)
                        Y = trail[j][-1]
                    else:
                        X = np.vstack((X, trail[j:j + self.T, :-1].reshape(1, -1)))
                        Y = np.append(Y, trail[j][-1])
            else:
                for j in range(trail.shape[0] - self.T + 1):
                    X = np.vstack((X, trail[j:j + self.T, :-1].reshape(1, -1)))
                    Y = np.append(Y, trail[j][-1])

        print(X.shape, Y.shape)
        return X.astype(np.float32), Y.astype(np.int32)
    #@override
    def get_train_data(self):
        return self.segmentation(flag='train')

    #@override
    def get_test_data(self):
        return self.segmentation(flag='test')

    #@override
    def get_X_Y(self):
        return self.segmentation(flag='all')

class SEED_IV_DATASET(Dataset):
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
    individual = 3
    session = 3
    balance = False
    modal = 'concat'
    # seed_iv = SEED_IV_SEG(individual=individual,T_len=9, session=session, modal=modal, balance=balance,shuffle=False)
    seed_iv = SEED_IV(individual=individual, session=session, modal=modal, balance=balance, shuffle=False, k_fold=6)
    # X, Y = seed_iv.get_X_Y()
    # print("X shape {} Y shape {}".format(X.shape, Y.shape))
    k_fold_data = seed_iv.get_training_kfold_data()
    for i, (train_X, train_Y, test_X, test_Y) in enumerate(k_fold_data):
        print("{}-th CV\t train X shape {}\n".format(i, train_X.shape))
        print("{}-th CV\t train Y shape {}\n".format(i, train_Y.shape))
        print("{}-th CV\t test X shape {}\n".format(i, test_X.shape))
        print("{}-th CV\t test Y shape {}\n".format(i, test_Y.shape))
        print("train Y == 0\t{}".format(sum(train_Y==0)))
        print("train Y == 1\t{}".format(sum(train_Y == 1)))
        print("train Y == 2\t{}".format(sum(train_Y == 2)))
        print("train Y == 3\t{}".format(sum(train_Y == 3)))
        print("test Y == 0\t{}".format(sum(test_Y == 0)))
        print("test Y == 1\t{}".format(sum(test_Y == 1)))
        print("test Y == 2\t{}".format(sum(test_Y == 2)))
        print("test Y == 3\t{}".format(sum(test_Y == 3)))

    # print("train X shape {}\ntrain Y shape {}\ntest X shape {}\ntest Y shape {}\n".format(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape))

    # train_X, train_Y = seed_iv.get_train_data()
    # test_X, test_Y = seed_iv.get_test_data()
    # print(train_X.shape)
    # print(train_Y.shape)
    # print(test_X.shape)
    # print(test_Y.shape)
    # X, Y = seed_iv.get_X_Y()
    # for i in np.unique(Y):
    #     print('label:{}, samples num:{}'.format(i, sum(Y==i)))
    # print(X.shape, Y.shape)
    # seed_iv_train_dataset = SEED_IV_DATASET(X=train_X, Y=train_Y)
    # seed_iv_test_dataset = SEED_IV_DATASET(X=test_X, Y=test_Y)
    # seed_iv_train_loader = DataLoader(dataset=seed_iv_train_dataset, shuffle=False,batch_size=32,num_workers=4)
    # seed_iv_test_loader = DataLoader(dataset=seed_iv_test_dataset, shuffle=True, batch_size=32,num_workers=4)
    # print(len(seed_iv_test_loader))
    # print(type(len(seed_iv_test_loader)))
    #
    # for i, (features, target) in enumerate(seed_iv_train_loader):
    #     print("batch %d" % (i+1))
    #     print(features.shape)
    #     print(target.shape)
    #     print(type(target))
    # for i, (features, target) in enumerate(seed_iv_test_loader):
    #     print("batch %d" % (i+1))
    #     print(features.shape)
    #     print(target.shape)
    #     print(type(target))

if __name__ == '__main__':
    main()




