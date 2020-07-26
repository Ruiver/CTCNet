"""
读取matlab的.mat格式的数据文件
date:2019/11/13
author:ouyangtianxiong
"""
#! -*-coding:utf-8-*-

import scipy.io as scio
import pickle
import codecs
import numpy as np
import os
import pyecharts.charts as Bar



class SEED_IV_PROCESOR:
    """
    SEED_IV 数据集构造函数
    眼动和EEG两种模态
    """
    def __init__(self):
        self.DATA_BASE_PATH = '../../multimodal_data/SEED_IV/'
        self.EEG_PATH = '../../multimodal_data/SEED_IV/eeg_feature_smooth'
        self.EYE_PATH = '../../multimodal_data/SEED_IV/eye_feature_smooth'
        self.SAVE_PATH = '../../multimodal_feature_extracted/SEED_IV'
    def merge_one_experiment(self, experiment_path, labels):
        """
        将SEED_IV中的每一次实验的脑电数据和眼动数据都融合起来
        :param experiment_path: 实验的子文件夹 [1,2,3]
                labels: 实验的刺激素材的情感极性的标签,一个24维的列表
        :return: 融合的数据应该是eeg+eye * sample * trails ,其中eeg的初始形状为
                62*sample*波段。转成62*5*sample 310 * sample 和  31 * sample的
                眼动特征融合成343维的特征
                返回一个字典，字典以被试名为键，它在某次实验中的脑电信号，眼动信号融合的24片段的
                二维numpy array为值。sample * 342 342= [310 + 31 + 1]
        """
        # 定义一个字典存储所有被试的数据
        # 定义该字典的键名，{subjectName_date}如 1_20180104.mat
        subject_feature_keys = os.listdir(os.path.join(self.EEG_PATH, experiment_path))
        subject_feature_dicts = {}
        # 定义eeg数据和eye数据的基路径
        base_eeg_path = os.path.join(os.path.join(self.EEG_PATH, experiment_path))
        base_eye_base = os.path.join(os.path.join(self.EYE_PATH, experiment_path))
        for key in subject_feature_keys:
            eeg_data = scio.loadmat(os.path.join(base_eeg_path, key))
            eye_data = scio.loadmat(os.path.join(base_eye_base, key))
            # 一共24个刺激片段
            subject_data_within_one_session = []
            for i in range(1, 25):
                eeg_trial_name = "de_LDS%d" % i
                eye_trial_name = "eye_%d" % i
                eeg_trial_data = eeg_data[eeg_trial_name]
                eye_trial_data = eye_data[eye_trial_name]
                eeg_trial_data = np.array(eeg_trial_data, dtype=np.float32)
                eye_trial_data = np.array(eye_trial_data, dtype=np.float32)
                assert eeg_trial_data.shape[0] == 62 and eeg_trial_data.shape[-1] == 5, "eeg数据维度错误"
                assert eye_trial_data.shape[0] == 31, "eye 数据特征维度错误"
                assert eeg_trial_data.shape[1] == eye_trial_data.shape[1], "eeg和eye样本维度对齐错误"
                data_labels = np.array([labels[i-1]] * eeg_trial_data.shape[1], dtype=np.int32)
                # print(data_labels)
                data_labels = data_labels[:, np.newaxis]
                # (n_electrode, sample, feature_dim) -> (sample,n_electrode,feature_dim)
                eeg_trial_data = eeg_trial_data.transpose((1, 0, 2)).reshape(-1, 310)
                data_temp = np.hstack((eeg_trial_data, eye_trial_data.transpose((1, 0)),data_labels))
                #self.statistic_data(data_temp)
                subject_data_within_one_session.append(data_temp)
            self.statistic_data(subject_data_within_one_session)
            subject_feature_dicts[key] = subject_data_within_one_session
        # 定义融合了的特征存储
        if not os.path.exists(os.path.join(self.SAVE_PATH, experiment_path)):
            os.mkdir(os.path.join(self.SAVE_PATH, experiment_path))

        with codecs.open(os.path.join(os.path.join(self.SAVE_PATH, experiment_path), 'fusion_feature.pk'), mode='wb') as f:
            pickle.dump(subject_feature_dicts, f)
        print("experiment %s process successful...")

    def load_data(self):
        experiment_path = os.listdir(self.EEG_PATH)
        session1_label = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                          [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                          [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
        for path in experiment_path:
            print("Dealing with experiment %s" % path)
            self.merge_one_experiment(path, session1_label[int(path)-1])
            print("Done with experiment %s" % path)
    def statistic_data(self,data):
        """统计数据集的样本数，分类数，各类别的样本数量
        :param data:  需要统计的数据，需要是一个numpy array
        :return:
        """
        tag = ['neutral', 'sad', 'fear', 'happy']
        assert isinstance(data,list), "请传入二维的数组"
        for i, arr in enumerate(data):
            if i == 0:
                arr_data = arr
            else:
                arr_data = np.vstack((arr_data, arr))
        data = arr_data
        data_num = data.shape[0]
        print("共 %d 条样本" % data_num)
        # 取最后一列是标签列
        labels = np.array(data[:, -1], dtype=np.int)
        unique_label = np.unique(labels)
        for l in unique_label:
            print("类别标签为 %s 的样本数共有 %d 个" % (tag[l], sum(labels == l)))

class DeapProcessor:
    def __init__(self):
        self.DATA_BASE_PATH = '../../multimodal_data/DEAP/DEAP-Mutlimodal-Features/'
        # self.DATA_BASE_PATH = '../../multimodal_data/DEAP/data_preprocessed_python/'
        self.DATA_SAVE_PATH = '../../multimodal_feature_extracted/DEAP'

    def show_data_detail(self):
        data = scio.loadmat(os.path.join(self.DATA_BASE_PATH, 's11.mat'))

        print(data.keys())
        for k, v in data.items():
            if k not in ['de_fea_used', 'new_label','label_all','time_fea_used']:
                continue
            print(k)
            print(v.shape)
            print(v[:3])
        #print(data['new_label'])
        #print(data['label_all'])


def main():
    procesor = DeapProcessor()
    procesor.show_data_detail()

if __name__ == '__main__':
    main()


