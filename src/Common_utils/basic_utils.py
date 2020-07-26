import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
def fill_ndarray(t1):             # 定义一个函数，把数组中为零的元素替换为一列的均值
    for i in range(t1.shape[1]):
        temp_col = t1[:,i]               # 取出当前列
        nan_num = np.count_nonzero(temp_col != temp_col)          # 判断当前列中是否含nan值
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()          # 用其余元素的均值填充nan所在位置
    return t1

def seed_normalization(train_X, train_Y, test_X, testY, nor_method=0, merge=0, column=0):
    """
    0 for minmax 1 for standard, 2 for nothing
    :param nor_method:
    :param merge:是否训练集测试集一起归一化
    :return:
    """
    # imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp_mean = KNNImputer(n_neighbors=10,weights="uniform")
    train_X = imp_mean.fit_transform(train_X)
    test_X = imp_mean.fit_transform(test_X)
    if column == 0:
        if nor_method == 0:
            scaler = MinMaxScaler()
        elif nor_method == 1:
            scaler = StandardScaler()
        elif nor_method == 2:
            scaler = Normalizer()
        elif nor_method == 3:
            scaler = Pipeline([('min_max', MinMaxScaler()),
                               ('standard', StandardScaler())])
        else:
            return train_X, train_Y, test_X, testY
        if merge == 0:
            scaler.fit(np.vstack((train_X, test_X)))
            train_X = scaler.transform(train_X)
            test_X = scaler.transform(test_X)
        elif merge == 1:
            scaler.fit(train_X)
            train_X = scaler.transform(train_X)
            test_X = scaler.transform(test_X)
        else:
            train_X = scaler.fit_transform(train_X)
            test_X = scaler.fit_transform(test_X)
        #scaler.fit(np.vstack((train_X, test_X)))
        return train_X, train_Y, test_X, testY
    else:
        train_X = train_X.T
        x_mean = np.mean(train_X, axis=0)
        x_std = np.std(train_X, axis=0)
        train_X = (train_X - x_mean) / (x_mean - x_std)

        test_X = test_X.T
        x_mean = np.mean(test_X, axis=0)
        x_std = np.std(test_X, axis=0)
        test_X = (test_X - x_mean) / (x_mean - x_std)

        return train_X.T, train_Y, test_X.T, testY
        
def deap_normalization(train_X, train_Y, test_X, testY, nor_method=0, merge=0, column=0):
    """
    0 for minmax 1 for standard, 2 for nothing
    :param nor_method:
    :param merge:是否训练集测试集一起归一化
    :return:
    """
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    train_X = imp_mean.fit_transform(train_X)
    test_X = imp_mean.fit_transform(test_X)
    if column == 0:
        if nor_method == 0:
            scaler = MinMaxScaler()
        elif nor_method == 1:
            scaler = StandardScaler()
        elif nor_method == 2:
            scaler = Normalizer()
        elif nor_method == 3:
            scaler = Pipeline([('min_max', MinMaxScaler()),
                               ('standard', StandardScaler())])
        else:
            return train_X, train_Y, test_X, testY
        if merge == 0:
            scaler.fit(np.vstack((train_X, test_X)))
            train_X = scaler.transform(train_X)
            test_X = scaler.transform(test_X)
        elif merge == 1:
            scaler.fit(train_X)
            train_X = scaler.transform(train_X)
            test_X = scaler.transform(test_X)
        else:
            train_X = scaler.fit_transform(train_X)
            test_X = scaler.fit_transform(test_X)
        #scaler.fit(np.vstack((train_X, test_X)))
        return train_X, train_Y, test_X, testY
    else:
        train_X = train_X.T
        x_mean = np.mean(train_X, axis=0)
        x_std = np.std(train_X, axis=0)
        train_X = (train_X - x_mean) / (x_mean - x_std)

        test_X = test_X.T
        x_mean = np.mean(test_X, axis=0)
        x_std = np.std(test_X, axis=0)
        test_X = (test_X - x_mean) / (x_mean - x_std)

        return train_X.T, train_Y, test_X.T, testY
