"""
author: ouyangtianxiong
date: 2020/3/03
des: implements attention-based emotion recognition on deap dataset
Based on code from https://github.com/KaihuaTang/VQA2.0-Recent-Approachs-2018.pytorch
"""
import sys
sys.path.append('../')
__author__ = 'ouyangtianxiong.bupt.edu.cn'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam,SGD,RMSprop
from torch.nn import CrossEntropyLoss
import numpy as np
from Common_utils.model_evaluation import plot_acc_loss_curve
from Common_utils.model_training import GradualWarmupScheduler, LabelSmoothSoftmax
from Common_utils.basic_module import FCNet
import os
from data_set.deap_feature import DEAP, DEAP_DATASET, DEAP128
#from Hierarchical_Attn import MultiBlocks, OneSideInterModalityUpdate, InterModalityUpdate,SingleBlock, Classifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from Common_utils.basic_utils import deap_normalization
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        # define number of detector for each sentiment class
        self.lin1 = FCNet(in_features, mid_features, activate='relu', drop=drop)
        self.lin2 = FCNet(mid_features, out_features, drop=drop)
        #
        self.bilinear = nn.Bilinear(in1_features=in_features, in2_features=in_features, out_features=mid_features)
    def forward(self, v, q):
        """
        :param v: [batch, r1, features]
        :param q: [batch, r2, features]
        :return:
        """
        num_obj = v.shape[2]
        max_len = q.shape[2]

        v_mean = v.sum(1) / num_obj
        q_mean = q.sum(1) / max_len
        #print("classifier v_mean", v_mean[0])
        #print("classifier q_mean", q_mean[0])

        #out = self.lin1(v_mean * q_mean)
        out = self.lin1(v_mean * q_mean)
        out = self.bilinear(v_mean, q_mean)
        #print("classifier out 1", out[0])
        out = self.lin2(out)
        #print("classifier out 2", out[0])
        return out
class InterModalityUpdate(nn.Module):
    """
    Inter-Modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(InterModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v_lin = FCNet(v_size, output_size * 3, drop=drop, activate='relu')
        self.q_lin = FCNet(q_size, output_size * 3, drop=drop, activate='relu')

        self.v_output = FCNet(output_size + v_size, output_size, drop=drop, activate='relu')
        self.q_output = FCNet(output_size + q_size, output_size, drop=drop, activate='relu')

    def forward(self, v, q):
        """
        :param v: eeg feature [batch, regions, feature_size]
        :param q: eye feature [batch, regions, feature_size]
        :return:
        """
        batch_size, num_obj = v.shape[0], v.shape[1]
        max_len = q.shape[1]

        # transfer feature to Q, K ,V matrix, here Q, K, V are concat together
        v_tran = self.v_lin(v)
        q_tran = self.q_lin(q)
        # mask all padding object/word feature
        # split Q, K, V
        v_key, v_query, v_val = torch.split(v_tran, v_tran.size(2) // 3, dim=2)
        q_key, q_query, q_val = torch.split(q_tran, q_tran.size(2) // 3, dim=2)

        # apply multi-head
        v_key_set = torch.split(v_key, v_key.size(2) // self.num_head, dim=2)
        v_query_set = torch.split(v_query, v_query.size(2) // self.num_head, dim=2)
        v_val_set = torch.split(v_val, v_val.size(2) // self.num_head, dim=2)
        q_key_set = torch.split(q_key, q_key.size(2) // self.num_head, dim=2)
        q_query_set = torch.split(q_query, q_query.size(2) // self.num_head, dim=2)
        q_val_set = torch.split(q_val, q_val.size(2) // self.num_head, dim=2)

        # apply multi-head operation
        for i in range(self.num_head):
            v_key_slice, v_query_slice, v_val_slice = v_key_set[i], v_query_set[i], v_val_set[i]
            q_key_slice, q_query_slice, q_val_slice = q_key_set[i], q_query_set[i], q_val_set[i]
            # calculating attention
            # [batch, num_obj, max_len]
            #print(v_query_slice.shape, q_key_slice.shape)
            q2v = (v_query_slice @ q_key_slice.transpose(1, 2)) / ((self.output_size // self.num_head) ** 0.5)
            #print(q_query_slice.shape, v_key_slice.shape)
            v2q = (q_query_slice @ v_key_slice.transpose(1, 2)) / ((self.output_size // self.num_head) ** 0.5)
            # softmax attention
            interMAF_q2v = F.softmax(q2v, dim=2).unsqueeze(3) #[batch_size, num_obj, max_len, 1]
            interMAF_v2q = F.softmax(v2q, dim=2).unsqueeze(3) #[batch_size, max_len, num_obj, 1] torch.cat((v_update, (interMAF_q2v * q_val_slice.unsqueeze(1)).sum(2)), dim=2)
            v_update = (interMAF_q2v * q_val_slice.unsqueeze(1)).sum(2) if (i == 0) else  torch.cat((v_update, (interMAF_q2v * q_val_slice.unsqueeze(1)).sum(2)), dim=2)
            q_update = (interMAF_v2q * v_val_slice.unsqueeze(1)).sum(2) if (i == 0) else torch.cat((q_update, (interMAF_v2q * v_val_slice.unsqueeze(1)).sum(2)), dim=2)
        # update new feature
        cat_v = torch.cat((v, v_update), dim=2)
        cat_q = torch.cat((q, q_update), dim=2)
        updated_v = self.v_output(cat_v)
        updated_q = self.q_output(cat_q)
        return updated_v, updated_q

class OneSideInterModalityUpdate(nn.Module):
    """
    one-side Inter-Modality Attention Flow
    according to the paper, instead of parallel V->Q & Q->V, we first to V->Q and then Q->V
    """
    def __init__(self,src_size,tgt_size,output_size,num_head,drop=0.0):
        super(OneSideInterModalityUpdate, self).__init__()
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.output_size = output_size
        self.num_head = num_head

        self.src_lin = FCNet(src_size, output_size * 2, drop=drop, activate='relu')
        self.tgt_lin = FCNet(tgt_size, output_size, drop=drop, activate='relu')

        self.tgt_output = FCNet(output_size + tgt_size, output_size, drop=drop, activate='relu')

    def forward(self, src, tgt):
        """
        :param src: eeg feature [batch, regions, feature_size]
        :param tgt: eye feature [batch, regions, feature_size]
        :return:
        """
        batch_size, num_src = src.shape[0],src.shape[1]
        num_tgt = tgt.shape[1]

        src_tran = self.src_lin(src)
        tgt_tran = self.tgt_lin(tgt)


        src_key, src_val = torch.split(src_tran, src_tran.size(2) // 2, dim=2)
        tgt_query = tgt_tran
        src_key_set = torch.split(src_key, src_key.size(2) // self.num_head, dim=2)
        src_val_set = torch.split(src_val, src_val.size(2) // self.num_head, dim=2)
        tgt_query_set = torch.split(tgt_query,tgt_query.size(2) // self.num_head, dim=2)
        for i in range(self.num_head):
            src_key_slice, tgt_query_slice, src_val_slice = src_key_set[i], tgt_query_set[i], src_val_set[i]
            src2tgt = (tgt_query_slice @ src_key_slice.transpose(1, 2)) / ((self.output_size // self.num_head) ** 0.5)
            interMAF_src2tgt = F.softmax(src2tgt, dim=2).unsqueeze(3)
            tgt_update = (interMAF_src2tgt * src_val_slice.unsqueeze(1)).sum(2) if (i == 0) else torch.cat((tgt_update, (interMAF_src2tgt * src_val_slice.unsqueeze(1)).sum(2)), dim=2)
        cat_tgt = torch.cat((tgt, tgt_update), dim=2)
        tgt_updated = self.tgt_output(cat_tgt)
        return tgt_updated

class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-Modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v4q_gate_lin = FCNet(v_size, output_size, drop=drop)
        self.q4v_gate_lin = FCNet(q_size, output_size, drop=drop)

        self.v_lin = FCNet(v_size, output_size * 3, drop=drop, activate='relu')
        self.q_lin = FCNet(q_size, output_size * 3, drop=drop, activate='relu')

        self.v_output = FCNet(output_size, output_size,drop=drop, activate='relu')
        self.q_output = FCNet(output_size, output_size, drop=drop, activate='relu')

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, v, q):
        """
        :param v: [batch_size, num_obj, feature_size]
        :param q: [batch_size, max_len, feature_size]

        :return:
        """
        batch_size, num_obj = v.shape[0], v.shape[1]
        max_len = q.shape[1]

        # conditioned gating vector
        v_mean = v.sum(1) / num_obj
        q_mean = q.sum(1) / max_len

        v4q_gate = self.sigmoid(self.v4q_gate_lin(v_mean)).unsqueeze(1) # [batch_size, 1, feature_size]
        q4v_gate = self.sigmoid(self.q4v_gate_lin(q_mean)).unsqueeze(1) # [batch_size, 1, feature_size]

        # K, Q, V
        v_tran = self.v_lin(v)
        q_tran = self.q_lin(q)

        # split for different use
        v_key, v_query, v_val = torch.split(v_tran, v_tran.size(2) // 3, dim=2)
        q_key, q_query, q_val = torch.split(q_tran, q_tran.size(2) // 3, dim=2)

        # apply conditioned gate
        gated_v_query = (1 + q4v_gate) * v_query
        gated_v_key = (1 + q4v_gate) * v_key
        gated_v_val = (1 + q4v_gate) * v_val
        gated_q_query = (1 + v4q_gate) * q_query
        gated_q_key = (1 + v4q_gate) * q_key
        gated_q_val = (1 + v4q_gate) * q_val

        # apply multi-head
        v_key_set = torch.split(gated_v_key, gated_v_key.size(2) // self.num_head, dim=2)
        v_query_set = torch.split(gated_v_query, gated_v_query.size(2) // self.num_head, dim=2)
        v_val_set = torch.split(gated_v_val, gated_v_val.size(2) // self.num_head, dim=2)
        q_key_set = torch.split(gated_q_key, gated_q_key.size(2) // self.num_head, dim=2)
        q_query_set = torch.split(gated_q_query, gated_q_query.size(2) // self.num_head, dim=2)
        q_val_set = torch.split(gated_q_val, gated_q_val.size(2) // self.num_head, dim=2)

        for i in range(self.num_head):
            v_key_slice, v_query_slice, v_val_slice = v_key_set[i], v_query_set[i], v_val_set[i]
            q_key_slice, q_query_slice, q_val_slice = q_key_set[i], q_query_set[i], q_val_set[i]
            # calcuating attention
            v2v = (v_query_slice @ v_key_slice.transpose(1,2)) / ((self.output_size // self.num_head) ** 0.5)
            q2q = (q_query_slice @ q_key_slice.transpose(1,2)) / ((self.output_size // self.num_head) ** 0.5)
            dyIntranMAF_v2v = F.softmax(v2v, dim=2).unsqueeze(3) # [batch_size, num_obj, num_obj, 1]
            dyIntranMAF_q2q = F.softmax(q2q, dim=2).unsqueeze(3) # [batch_size, max_len, max_len, 1]
            # calculating update input
            v_update = (dyIntranMAF_v2v * v_val_slice.unsqueeze(1)).sum(2) if (i == 0) else torch.cat((v_update, (dyIntranMAF_v2v * v_val_slice.unsqueeze(1)).sum(2)), dim=2)
            q_update = (dyIntranMAF_q2q * q_val_slice.unsqueeze(1)).sum(2) if (i == 0) else torch.cat((q_update, (dyIntranMAF_q2q * q_val_slice.unsqueeze(1)).sum(2)), dim=2)

        # update
        updated_v = self.v_output(v + v_update)
        updated_q = self.q_output(q + q_update)
        return updated_v, updated_q

class SingleBlock(nn.Module):
    """
        Single Block Inter- and Intra modality stack multiple times, in such circumstance, all the
        basic blocks share the same parameters in the model
    """
    def __init__(self, num_blocks, v_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(SingleBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_blocks

        self.v_lin = FCNet(v_size, output_size, drop=drop, activate='relu')
        self.q_lin = FCNet(q_size, output_size, drop=drop, activate='relu')

        self.v2q_interBlock = OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.q2v_interBlock = OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.intraBlock = DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop)

    def forward(self, v, q):
        """
        :param v: eeg feature [batch_size, regions, feature_size]
        :param q: eye feature [batch_size, regions, feature_size]
        :return:
        """
        # transfer features
        v = self.v_lin(v)
        q = self.q_lin(q)
        # residual connection
        v_container = [v]
        q_container = [q]
        result_v = [v]
        result_q = [q]
        for i in range(self.num_block):
            q1 = self.v2q_interBlock(v_container[-1], q_container[-1])
            q_container.append(q1)
            v1 = self.q2v_interBlock(q_container[-1], v_container[-1])
            v_container.append(v1)
            v2, q2 = self.intraBlock(v_container[-1] + v_container[-2], q_container[-1] + q_container[-2])
            v_container.append(v2)
            q_container.append(q2)
            result_v.append(v1)
            result_v.append(v2)
            result_q.append(q1)
            result_q.append(q2)
            v_container.append(v_container[-1] + v_container[-2] + v_container[-3])
            q_container.append(q_container[-1] + q_container[-2] + q_container[-3])
        return sum(result_v), sum(result_q)

class MultiBlocks(nn.Module):
    """
    Stack multiple single block layer, each layer possess their own parameters
    """

    def __init__(self, num_blocks, v_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(MultiBlocks, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_blocks = num_blocks

        self.v_lin = FCNet(v_size, output_size, drop=drop, activate='relu')
        self.q_lin = FCNet(q_size, output_size, drop=drop, activate='relu')

        blocks = []
        for i in range(self.num_blocks):
            #blocks.append(OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop))
            #blocks.append(OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop))
            blocks.append(InterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop))
            blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop))
        self.multi_blocks = nn.ModuleList(blocks)

    def forward(self, v, q):
        """
        :param v: eeg feature [batch, regions, feature_size]
        :param q: eye feature [batch, regions, feature_size]
        :return:
        """
        # transfer feature
        v = self.v_lin(v)
        q = self.q_lin(q)
        v_container = [v]
        q_container = [q]
        result_v = [v]
        result_q = [q]

        # dense residule connection
        for i in range(self.num_blocks):
            # q1 = self.multi_blocks[i * 3 + 0](v_container[-1], q_container[-1])
            # q_container.append(q1)
            # v1 = self.multi_blocks[i * 3 + 1](q_container[-1], v_container[-1])
            # v_container.append(v1)
            v1, q1 = self.multi_blocks[i * 2 + 0](v_container[-1], q_container[-1])
            q_container.append(q1)
            v_container.append(v1)
            v2, q2 = self.multi_blocks[i * 2 + 1](v_container[-1] + v_container[-2], q_container[-1] + q_container[-2])
            v_container.append(v2)
            q_container.append(q2)
            result_v.append(v1)
            result_v.append(v2)
            result_q.append(q1)
            result_q.append(q2)
            v_container.append(v_container[-1] + v_container[-2] + v_container[-3])
            q_container.append(q_container[-1] + q_container[-2] + q_container[-3])
        return sum(result_v), sum(result_q)
class EEGFeatureExtractor(nn.Module):
    def __init__(self, eeg_size, output_size):
        super(EEGFeatureExtractor, self).__init__()
        self.eeg_size = eeg_size
        self.output_size = output_size
        self.regions = 14  # regions的数量
        self.regions_indexs = [torch.LongTensor(e) for e in
                               [[0, 1, 16, 17], [2, 18, 19], [3, 4], [20], [7,8],
                                [21,25,26], [5, 22], [6,23,24],
                                [9,27], [11], [29], [10,15,28],
                                [12, 30], [13, 14,31]]]
        reginal_extractors = []
        for i in range(self.regions):
            reginal_extractors.append(nn.LSTM(input_size=eeg_size, hidden_size= output_size // 2, batch_first=True, bias=True, bidirectional=True))

        self.reginalFeatureExtractors = nn.ModuleList(reginal_extractors)
        self.bn = nn.BatchNorm1d(num_features=self.regions)

    def forward(self, x):
        """
        :param x: [batch, n_electrode, 128]
        :return: [batch, regions, feature_size]
        """
        batch, n_electrode, _ = x.shape
        X_regions_input = []  # 列表存储不同区域的张量输入
        for i in range(self.regions):
            X_regions_input.append(x.index_select(dim=1, index=self.regions_indexs[i].to(device)))
        X_regional_lstm_out = []
        for i in range(self.regions):
            shape = X_regions_input[i].shape
            # print(shape)
            # 先转成（B*T,n_i,d）再进LSTM
            hidden_units, _ = self.reginalFeatureExtractors[i](X_regions_input[i].reshape((-1, shape[-2], shape[-1])))
            X_regional_lstm_out.append(hidden_units[:, -1, :].squeeze())
        # X_regional_feature : 列表：元素为tensor [ B*T, regions_num, 2*self.d_r]
        # reshape成(B*T, regions, 2*self.d_r)
        # (B * T, regions, 2* self.d_r)
        X_regional_feature = torch.cat(X_regional_lstm_out, dim=-1).reshape(batch, self.regions, self.output_size)
        return self.bn(X_regional_feature)


class PeripheralFeatureExtractor(nn.Module):
    def __init__(self, peripheral_size, output_size):
        super(PeripheralFeatureExtractor, self).__init__()
        self.peripheral_size = peripheral_size
        self.output_size = output_size
        self.regions = 8
        self.regions_indexs = [torch.LongTensor(e) for e in
                               [[0],
                                [1],
                                [2],
                                [3],
                                [4],
                                [5],
                                [6],
                                [7]]]
        eye_extractor = []
        eye_extractor.append(nn.LSTM(input_size=peripheral_size, hidden_size= output_size // 2, batch_first=True, bias=True, bidirectional=True))
        eye_extractor.append(nn.LSTM(input_size=peripheral_size, hidden_size= output_size // 2, batch_first=True, bias=True, bidirectional=True))
        eye_extractor.append(nn.LSTM(input_size=peripheral_size, hidden_size= output_size // 2, batch_first=True, bias=True, bidirectional=True))
        eye_extractor.append(nn.LSTM(input_size=peripheral_size, hidden_size= output_size // 2, batch_first=True, bias=True, bidirectional=True))
        eye_extractor.append(nn.LSTM(input_size=peripheral_size, hidden_size= output_size // 2, batch_first=True, bias=True, bidirectional=True))
        eye_extractor.append(nn.LSTM(input_size=peripheral_size, hidden_size= output_size // 2, batch_first=True, bias=True, bidirectional=True))
        eye_extractor.append(nn.LSTM(input_size=peripheral_size, hidden_size= output_size // 2, batch_first=True, bias=True, bidirectional=True))
        eye_extractor.append(nn.LSTM(input_size=peripheral_size, hidden_size= output_size // 2, batch_first=True, bias=True, bidirectional=True))
        self.eyeFeatureExtractor = nn.ModuleList(eye_extractor)
        self.bn = nn.BatchNorm1d(num_features=self.regions)

    def forward(self, x):
        """
        :param x: EYE feature [batch, peripheral_num, 128]
        :return: [batch, regons, output_size]
        """
        batch, n_electrode, _ = x.shape
        X_regions_input = []  # 列表存储不同区域的张量输入
        for i in range(self.regions):
            X_regions_input.append(x.index_select(dim=1, index=self.regions_indexs[i].to(device)))
        X_regional_lstm_out = []
        for i in range(self.regions):
            shape = X_regions_input[i].shape
            # print(shape)
            # 先转成（B*T,n_i,d）再进LSTM
            hidden_units, _ = self.eyeFeatureExtractor[i](X_regions_input[i].reshape((-1, shape[-2], shape[-1])))
            X_regional_lstm_out.append(hidden_units[:, -1, :].squeeze())
        # X_regional_feature : 列表：元素为tensor [ B*T, regions_num, 2*self.d_r]
        # reshape成(B*T, regions, 2*self.d_r)
        # (B * T, regions, 2* self.d_r)
        X_regional_feature = torch.cat(X_regional_lstm_out, dim=-1).reshape(batch, self.regions, self.output_size)
        return self.bn(X_regional_feature)

class PeripheralFeatureExtractor2(nn.Module):
    def __init__(self, peripheral_size, output_size):
        super(PeripheralFeatureExtractor2, self).__init__()
        self.peripheral_size = peripheral_size
        self.output_size = output_size
        self.regions = 6
        self.regions_indexs = [torch.LongTensor(e) for e in
                               [[0, 1],
                                [2, 3],
                                [4],
                                [5],
                                [6],
                                [7]]]
        eye_extractor = []
        eye_extractor.append(FCNet(in_size=10, out_size=output_size, activate='relu'))
        eye_extractor.append(FCNet(in_size=10, out_size=output_size, activate='relu'))
        eye_extractor.append(FCNet(in_size=5, out_size=output_size, activate='relu'))
        eye_extractor.append(FCNet(in_size=5, out_size=output_size, activate='relu'))
        eye_extractor.append(FCNet(in_size=5, out_size=output_size, activate='relu'))
        eye_extractor.append(FCNet(in_size=5, out_size=output_size, activate='relu'))
        self.eyeFeatureExtractor = nn.ModuleList(eye_extractor)
        self.bn = nn.BatchNorm1d(num_features=self.regions)

    def forward(self, x):
        """
        :param x: peripheral feature [batch, 8, 5]
        :return: [batch, regions, output_size]
        """
        B = x.shape[0]
        X_regions_input = []  # 列表存储不同区域的张量输入
        for i in range(self.regions):
            tmp = x.index_select(dim=1, index=self.regions_indexs[i].to(device))
            #print(tmp.shape)
            tmp = tmp.reshape(B, -1)
            #print(tmp.shape)
            X_regions_input.append(tmp)
        X_regional_output = []
        for i in range(self.regions):
            X_regional_output.append(
                self.eyeFeatureExtractor[i](X_regions_input[i])
            )
        X_regional_feature = torch.cat(X_regional_output, dim=-1).reshape(B, self.regions, self.output_size)
        return self.bn(X_regional_feature)

class Senti_Map_Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Senti_Map_Classifier, self).__init__()
        # define number of detector for each sentiment class
        self.k = 10
        self.emotion_class = out_features
        eeg_detectors = []
        eye_detectors = []
        for i in range(out_features):
            eeg_detectors.append(
                nn.Conv1d(in_channels=in_features, out_channels=self.k, kernel_size=1, stride=1, padding=0, bias=True))
            eye_detectors.append(
                nn.Conv1d(in_channels=in_features, out_channels=self.k, kernel_size=1, stride=1, padding=0, bias=True))
        self.eeg_detectors = nn.ModuleList(eeg_detectors)
        self.eye_detectors = nn.ModuleList(eye_detectors)

        self.lin1 = FCNet(in_features * self.emotion_class, mid_features, activate='relu', drop=drop)
        self.lin2 = FCNet(mid_features, out_features, drop=drop)
        emotion_classifer = []
        self.bilinears = nn.ModuleList([nn.Bilinear(in1_features=in_features, in2_features=in_features, out_features=in_features) for _ in range(self.emotion_class)])
        for i in range(self.emotion_class):
            emotion_classifer.append(nn.Sequential(
                nn.Dropout(p=drop),
                FCNet(in_features, mid_features, activate='relu', drop=drop),
                FCNet(mid_features, 1, drop=drop)
            ))
        self.emotion_classifer = nn.ModuleList(emotion_classifer)

    def eeg_senti_relevance_detect(self, v):
        b, r, d = v.shape
        v = v.permute(0, 2, 1)
        eeg_activate = []
        for i in range(self.emotion_class):
            # [batch, k, r]
            k_eeg_activate_per_class = torch.softmax(self.eeg_detectors[i](v), dim=-1)
            # [batch, 1, k ,r]
            eeg_activate = k_eeg_activate_per_class.unsqueeze(dim=1) if i == 0 else torch.cat(
                (eeg_activate, k_eeg_activate_per_class.unsqueeze(dim=1)), dim=1)
        assert eeg_activate.shape == torch.Size([b, self.emotion_class, self.k, r]), "sentiment map wrong!! {}".format(
            eeg_activate.shape)
        return eeg_activate.permute(0, 1, 3, 2)

    def eye_senti_relevance_detect(self, v):
        b, r, d = v.shape
        v = v.permute(0, 2, 1)
        eye_activate = []
        for i in range(self.emotion_class):
            # [batch, k, r]
            k_eye_activate_per_class = torch.softmax(self.eeg_detectors[i](v), dim=-1)
            # [batch, 1, k ,r]
            eye_activate = k_eye_activate_per_class.unsqueeze(dim=1) if i == 0 else torch.cat(
                (eye_activate, k_eye_activate_per_class.unsqueeze(dim=1)), dim=1)
        assert eye_activate.shape == torch.Size([b, self.emotion_class, self.k, r]), "sentiment map wrong!! {}".format(
            eye_activate.shape)
        return eye_activate.permute(0, 1, 3, 2)

    def forward(self, v, q):
        """
        :param v: [batch, r1, features]
        :param q: [batch, r2, features]
        :return:
        """
        b, r1 = v.shape[0], v.shape[1]
        r2 = q.shape[1]
        eeg = v
        eye = q
        # [batch, emotion_class, r1, k]
        eeg_senti_relevance = self.eeg_senti_relevance_detect(v)
        # [batch, emotion_class, r2, k]
        eye_senti_relevance = self.eye_senti_relevance_detect(q)

        # # [batch, emotion_class, r1,1]
        # attn_eeg = eeg_senti_relevance.sum(dim=3, keepdims=True) / self.k
        #
        # # [batch, emotion_class, r2,1]
        # attn_eye = eye_senti_relevance.sum(dim=3, keepdims=True) / self.k
        #
        # # [batch, emotion_class, r1, 1] * [batch, 1, r1, features] = [batch, emotion class, r, features]
        # # introduce learnable sentiment relevance
        # map_eeg = attn_eeg * eeg.unsqueeze(dim=1)
        # map_eye = attn_eye * eye.unsqueeze(dim=1)
        # # [batch, emotion_class, feature]
        # map_eeg = map_eeg.sum(dim=2) / r1
        # map_eye = map_eye.sum(dim=2) / r2
        #
        #
        # fusion_feature = map_eeg * map_eye
        # final = fusion_feature.view(b, -1)
        # out = self.lin1(final)
        # out = self.lin2(out)
        # [batch, emotion_class, r1,1]
        attn_eeg = eeg_senti_relevance.sum(dim=3, keepdim=True) / self.k
        emotion_eeg_attn = attn_eeg.squeeze().sum(dim=2, keepdim=False) / attn_eeg.size(2)
        # [batch, emotion_class] represents the maximum activate in sentiment map
        emotion_eeg_attn = emotion_eeg_attn.squeeze()
        # [batch, emotion_class, r2,1]
        attn_eye = eye_senti_relevance.sum(dim=3, keepdim=True) / self.k
        emotion_eye_attn = attn_eye.squeeze().sum(dim=2, keepdim=False) / attn_eye.size(2)
        emotion_eye_attn = emotion_eye_attn.squeeze()

        # [batch, emotion_class, r1, 1] * [batch, 1, r1, features] = [batch, emotion class, r, features]
        # introduce learnable sentiment relevance
        map_eeg = attn_eeg * eeg.unsqueeze(dim=1)
        map_eye = attn_eye * eye.unsqueeze(dim=1)

        out = []
        for i in range(self.emotion_class):
            emotion_specific_eeg = map_eeg[:, i, :, :].squeeze().mean(1).squeeze()
            emotion_specific_eye = map_eye[:, i, :, :].squeeze().mean(1).squeeze()
            tmp = self.bilinears[i](emotion_specific_eeg, emotion_specific_eye)
            out1 = self.emotion_classifer[i](tmp)
            out.append(out1)
        out = torch.cat(out, dim=-1)
        return out # , torch.softmax(emotion_eeg_attn,dim=1), torch.softmax(emotion_eye_attn,dim=1)

class Hierarchical_ATTN_With_Senti_Map(nn.Module):
    def __init__(self, class_num=4):
        super(Hierarchical_ATTN_With_Senti_Map, self).__init__()
        # self.eeg_features = 16 # 256
        # self.peripheral_features = 16 # 256
        self.eeg_features = 64  # 256
        self.peripheral_features = 64  # 256
        self.hidden_feature = 128 # 256
        self.num_inter_head = 4
        self.num_intra_head = 4
        self.num_block = 1

        assert self.hidden_feature % self.num_inter_head == 0, 'hidden features size can not be divided by header nums, please check!!'
        assert self.hidden_feature % self.num_inter_head == 0, 'hidden features size can not be divided by header nums, please check!!'

        # basic feature extractor
        self.eegFeatureExtractor = EEGFeatureExtractor(eeg_size=5, output_size=self.eeg_features)

        self.eyeFeatureExtractor = PeripheralFeatureExtractor2(peripheral_size=5, output_size=self.peripheral_features)

        # inter- & intra-modality attention flow mechanism for fusion cross modality feature
        self.interIntraBlocks = MultiBlocks(
            num_blocks=self.num_block,
            v_size=self.peripheral_features,
            q_size=self.eeg_features,
            output_size=self.hidden_feature,
            num_inter_head=self.num_inter_head,
            num_intra_head=self.num_intra_head,
            drop=0.5
        )

        # emotion classifier
        # self.classifier = Classifier(
        #     in_features=self.hidden_feature,
        #     mid_features=512, out_features=class_num,
        #     drop=0.5)
        self.classifier = Senti_Map_Classifier(
            in_features=self.hidden_feature,
            mid_features=512, out_features=class_num,
            drop=0.5)
    def forward(self, v, q):
        """
        :param v: eeg feature [batch, n, 5]
        :param q:  eye feature [batch, 31]
        :return: predict logits [batch, max_answer]
        """
        # prepare v & q feature

        v = self.eegFeatureExtractor(v)
        q = self.eyeFeatureExtractor(q)

        # feature normalization
        v = v / (v.norm(p=2, dim=2, keepdim=True) + 1e-12).expand_as(v) # [batch, num_obj, feature]
        q = q / (q.norm(p=2, dim=2, keepdim=True) + 1e-12).expand_as(q)

        # inter- & intra- modality attention flow
        v, q = self.interIntraBlocks(v, q)

        # predict logits
        answer = self.classifier(v, q)
        return answer

def generate_k_data(data,n_split=10,shuffle=True):
    if shuffle:
        np.random.shuffle(data)
    total_count = data.shape[0]
    for k in range(n_split):
        pass



def subject_dependent(individual=1, class_target=4):
    class_list = [
        "Valence",
        "Arousal",
        "Dominance",
        "Liking",
        "Valence-Arousal"
    ]
    class_nums = [2,2,2,2,4]
    test_loss_list = []  # 记录每一折验证的loss
    test_acc_list = []  # 记录每一折验证的acc
    # prepare data
    nor_method = 1
    label_smooth = 0.1
    shuffle = True

    # reading the data in the whole dataset
    deap = DEAP(individual=individual, normalization=nor_method)
    train_X, train_Y = deap.get_train_data()
    validate_X, validate_Y = deap.get_validate_data()
    test_X, test_Y = deap.get_test_data()
    
    # Hyper-parameters
    epochs = 100
    batch_size = 512
    learning_rate = 5e-6
    criterion = LabelSmoothSoftmax(lb_smooth=label_smooth)
    # criterion_attn = CrossEntropyLoss()
    print("starting subject-dependent training experiments on individual %d class %s" % (
    individual, class_list[class_target]))

    print("train_X shape", train_X.shape)
    print("train_Y shape", train_Y.shape)
    print("validate_X shape", validate_X.shape)
    print("validate_Y shape", validate_Y.shape)
    print("test_X shape", test_X.shape)
    print("test_Y shape", test_Y.shape)

    train_Y, test_Y, validate_Y = train_Y[:, class_target].squeeze(), test_Y[:, class_target].squeeze(), validate_Y[:, class_target].squeeze()
    train_loader = DataLoader(dataset=DEAP_DATASET(train_X, train_Y), batch_size=batch_size, shuffle=shuffle,
                              num_workers=0)
    validate_loader = DataLoader(dataset=DEAP_DATASET(validate_X, validate_Y), batch_size=batch_size, shuffle=shuffle,
                              num_workers=0)
    test_loader = DataLoader(dataset=DEAP_DATASET(test_X, test_Y), batch_size=batch_size, shuffle=shuffle,
                             num_workers=0)

    exp_des = "%d_dependent_%s_%s_%d_%d_%s" % (
        individual, 'shuffle' if shuffle else "without_shuffle", 'deap', epochs, batch_size,
        class_list[class_target])

    print("model construction...")
    net = Hierarchical_ATTN_With_Senti_Map(class_num=class_nums[class_target])
    # if fine_tuning we continue train the pretrained model
    net = net.to(device)
    save_model_path = '../../saved_models/%s/deap/subject_%d/%s/' % (
    net.__class__.__name__, individual, class_list[class_target])

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    optimization = Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)

    # save model training state
    running_loss_list = []
    running_acc_list = []
    validate_loss_list = []
    validate_acc_list = []
    best_acc = -1
    print("start training...")
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimization, T_max=epochs)
    scheduler_warmup = GradualWarmupScheduler(optimizer=optimization, multiplier=10,
                                              total_epoch=np.ceil(0.1 * epochs),
                                              after_scheduler=scheduler_cosine)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (feature, target) in enumerate(train_loader):
            feature = feature.reshape(-1, 40, 5)
            optimization.zero_grad()
            #print("训练集label:{}".format(target))
            # print("脏数据统计", torch.sum(torch.isnan(feature), dim=0))
            eeg = feature[:, :32, :]
            peripheral = feature[:, 32:, :]
            eeg = eeg.reshape(-1, 32, 5)
            peripheral = peripheral.reshape(-1, 8, 5)
            eeg = eeg.to(device)
            peripheral = peripheral.to(device)
            target = target.type(torch.LongTensor).to(device)
            out = net(eeg, peripheral)
            #print("训练集", out.data[:5])
            #print("训练集",eeg_attn.shape, eeg_attn.data[:5])
            #print("训练集",eye_attn.shape, eye_attn.data[:5])
            # print("batch output",out[0])
            cross_entropy_loss = criterion(out, target)
            # eeg_attn_loss = criterion_attn(eeg_attn, target)
            # eye_attn_loss = criterion_attn(eye_attn, target)
            #loss = cross_entropy_loss
            #print("交叉熵损失", cross_entropy_loss.data)
            #print("eeg注意力损失", eeg_attn_loss.data)
            #print("eye注意力损失", eeg_attn_loss.data)
            cross_entropy_loss.backward()
            clip_grad_norm_(net.parameters(), max_norm=10)
            # for name, parms in net.named_parameters():
            #     print('打印梯度')
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
            #           ' -->grad_value:', parms.grad)
            optimization.step()
            running_loss += cross_entropy_loss.item()
            # print("batch loss", loss.item())
            _, prediction = torch.max(out.data, dim=-1)
            #print('训练集', prediction[:5])
            total += target.size(0)
            correct += prediction.eq(target.data).cpu().sum().item()
        cur_loss = running_loss / total
        cur_acc = correct / total
        # print(cur_acc, correct, total)
        if isinstance(cur_acc, torch.Tensor):
            cur_acc = cur_acc.item()
        if isinstance(cur_loss, torch.Tensor):
            cur_loss = cur_loss.item()
        print('Training Loss: %.10f | Training Acc: %.3f%% (%d/%d)' % (
            cur_loss, 100 * cur_acc, correct, total))
        running_loss_list.append(cur_loss)
        running_acc_list.append(cur_acc)
        scheduler_warmup.step()
        if epoch % 5 == 0:
            net.eval()
            print("start evaluating...")
            validate_loss = 0.0
            validate_correct = 0.0
            validate_total = 0.0
            for i, (feature, target) in enumerate(validate_loader):
                feature = feature.reshape(-1, 40, 5)
                #print("训练集label:{}".format(target))
                eeg = feature[:, :32, :]
                peripheral = feature[:, 32:, :]
                eeg = eeg.reshape(-1, 32, 5)
                peripheral = peripheral.reshape(-1, 8, 5)
                eeg = eeg.to(device)
                peripheral = peripheral.to(device)
                target = target.type(torch.LongTensor).to(device)
                with torch.no_grad():
                    out= net(eeg, peripheral)
                    #print("c集", out.data[:5])
                    #print("c集", eeg_attn.data[:5])
                    #print("c集", eye_attn.data[:5])
                    loss = criterion(out, target)
                    validate_loss += loss.item()
                    _, prediction = torch.max(out.data, dim=-1)
                    #print('验证集',prediction[:10])
                    validate_total += target.size(0)
                    validate_correct += prediction.eq(target.data).cpu().sum().item()
                    #print("验证集相等:{}".format(prediction.eq(target.data)))
            validate_acc = validate_correct / validate_total
            validate_loss = validate_loss / validate_total
            if isinstance(validate_acc, torch.Tensor):
                validate_acc = validate_acc.item()
            if isinstance(validate_loss, torch.Tensor):
                validate_loss = validate_loss.item()
            print('Validate Loss: %.10f | Validate-Acc: %.3f%% (%d/%d)' % (
                validate_loss, 100 * validate_acc, validate_correct, validate_total))
            validate_acc_list.append(validate_acc)
            validate_loss_list.append(validate_loss)
            if validate_acc > best_acc:
                best_acc = validate_acc
                print("better model founded in validating sets, start saving new model")
                model_name = '%s' % (net.__class__.__name__)
                state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'current_loss': validate_loss
                }
                torch.save(state, os.path.join(save_model_path, model_name))
    # 开始计算测试集
    checkpoint = torch.load(os.path.join(save_model_path, net.__class__.__name__))
    net.load_state_dict(checkpoint['net'])
    print("start evaluating...")
    testing_loss = 0.0
    test_correct = 0.0
    test_total = 0.0
    y_pre = []
    y_true = []
    for i, (feature, target) in enumerate(test_loader):
        feature = feature.reshape(-1, 40, 5)
        eeg = feature[:, :32, :]
        peripheral = feature[:, 32:, :]
        eeg = eeg.reshape(-1, 32, 5)
        peripheral = peripheral.reshape(-1, 8, 5)
        eeg = eeg.to(device)
        peripheral = peripheral.to(device)
        target = target.type(torch.LongTensor).to(device)
        y_true.extend(target.cpu().numpy().tolist())
        with torch.no_grad():
            out = net(eeg, peripheral)
            loss = criterion(out, target)
            testing_loss += loss.item()
            _, prediction = torch.max(out.data, dim=-1)
            y_pre.extend(prediction.cpu().numpy().tolist())
            # print(prediction)
            test_total += target.size(0)
            test_correct += prediction.eq(target.data).cpu().sum().item()
    test_acc = test_correct / test_total
    test_loss = testing_loss / test_total
    if isinstance(test_acc, torch.Tensor):
        test_acc = test_acc.item()
    if isinstance(test_loss, torch.Tensor):
        test_loss = test_loss.item()
    print('Test Loss: %.10f | Test Acc: %.3f%% (%d/%d)' % (
        test_loss, 100 * test_acc, test_correct, test_total))
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)
    plot_acc_loss_curve({'train_loss': running_loss_list,
                         'train_acc': running_acc_list,
                         'test_loss': validate_loss_list,
                         'test_acc': validate_acc_list}, net.__class__.__name__, exp_des)
    pd.DataFrame.from_dict({
        'test_loss': test_loss_list,
        'test_acc': test_acc_list
    }).to_csv('./results/deap_individual_%d_%s.csv' % (individual, class_list[class_target]), mode='w', index=False,
              header=True, encoding='utf-8')
    y_true = np.array(y_true)
    y_pre = np.array(y_pre)
    with open('./results/{}_classification_reports.txt'.format(class_list[class_target]), 'a+') as f:
        f.write("***********Predict results of individual {}***********\n".format(individual))

        f.write("classification reports:\n{}\nconfusion matrix:\n{}\noytx_accuracy_score:{}\noytx_precision_score:{}\noytx_recall_score:{}\noytx_f1_score:{}\n".format(classification_report(y_true, y_pre), confusion_matrix(y_true, y_pre),accuracy_score(y_true, y_pre),precision_score(y_true, y_pre, average='macro'),recall_score(y_true, y_pre, average='macro'),f1_score(y_true,y_pre, average='macro')))

        f.write("******************************************************\n")

def subject_independent(all_X, all_Y, individual=1, class_target=4):
    # 跨个体，留一法
    class_list = [
        "Valence",
        "Arousal",
        "Dominance",
        "Liking",
        "Valence-Arousal"
    ]
    class_nums = [2,2,2,2,4]
    test_loss_list = []  # 记录每一折验证的loss
    test_acc_list = []  # 记录每一折验证的acc
    # prepare data
    nor_method = 1
    label_smooth = 0.1
    shuffle = True
    kfold = 5

    # reading the data in the whole dataset
    # Hyper-parameters
    epochs = 120
    batch_size = 512
    learning_rate = 1e-3
    criterion = LabelSmoothSoftmax(lb_smooth=label_smooth)
    criterion_attn = CrossEntropyLoss()
    print("starting subject-independent training experiments on individual %d class %s" % (
    individual, class_list[class_target]))


    train_X = np.vstack([item for i, item in enumerate(all_X) if i != individual]).reshape(-1,40,128)
    train_Y = np.vstack([item for i, item in enumerate(all_Y) if i != individual]).reshape(-1,5)
    test_X = all_X[individual].reshape(-1,40,128)
    test_Y = all_Y[individual].reshape(-1,5)

    sample_index = list(range(train_X.shape[0]))
    if shuffle:
        np.random.seed(seed=0)
        np.random.shuffle(sample_index)
    val_X, val_Y = train_X[sample_index[:int(len(sample_index)*0.3)]], train_Y[sample_index[:int(len(sample_index)*0.3)]]
    #train_X, train_Y = train_X[sample_index[int(len(sample_index)*0.3):]], train_Y[sample_index[int(len(sample_index)*0.3):]]
    print("train_X shape", train_X.shape)
    print("train_Y shape", train_Y.shape)
    print("val_X shape", val_X.shape)
    print("val_Y shape", val_Y.shape)
    print("test_X shape", test_X.shape)
    print("test_Y shape", test_Y.shape)
    train_Y, test_Y, val_Y = train_Y[:, class_target].squeeze(), test_Y[:, class_target].squeeze(), val_Y[:, class_target].squeeze()
    train_loader = DataLoader(dataset=DEAP_DATASET(train_X, train_Y), batch_size=batch_size, shuffle=shuffle,
                              num_workers=0)
    val_loader = DataLoader(dataset=DEAP_DATASET(val_X, val_Y), batch_size=batch_size, shuffle=shuffle,
                              num_workers=0)
    test_loader = DataLoader(dataset=DEAP_DATASET(test_X, test_Y), batch_size=batch_size, shuffle=shuffle,
                             num_workers=0)

    exp_des = "%d_dependent_in_%s_%s_%d_%d_%s" % (
        individual, 'shuffle' if shuffle else "without_shuffle", 'deap', epochs, batch_size,
        class_list[class_target])
    print("model construction...")
    net = Hierarchical_ATTN_With_Senti_Map(class_num=class_nums[class_target])
    # if fine_tuning we continue train the pretrained model
    net = net.to(device)
    save_model_path = '../../saved_models/%s/deap_subjuect_independent/subject_%d/%s/' % (
    net.__class__.__name__, individual, class_list[class_target])
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    optimization = Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)

    # save model training state
    running_loss_list = []
    running_acc_list = []
    testing_loss_list = []
    testing_acc_list = []
    best_acc = -1
    print("start training...")
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimization, T_max=epochs)
    scheduler_warmup = GradualWarmupScheduler(optimizer=optimization, multiplier=10,
                                              total_epoch=np.ceil(0.1 * epochs),
                                              after_scheduler=scheduler_cosine)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (feature, target) in enumerate(train_loader):
            optimization.zero_grad()
            # print("脏数据统计", torch.sum(torch.isnan(feature), dim=0))
            eeg = feature[:, :32]
            peripheral = feature[:, 32:]
            eeg = eeg.reshape(-1, 32, 128)
            eeg = eeg.to(device)
            peripheral = peripheral.reshape(-1, 8, 128)
            peripheral = peripheral.to(device)
            target = target.type(torch.LongTensor).to(device)
            out, eeg_attn, eye_attn = net(eeg, peripheral)
            # print("batch output",out[0])
            cross_entropy_loss = criterion(out, target)
            eeg_attn_loss = criterion_attn(eeg_attn, target)
            eye_attn_loss = criterion_attn(eye_attn, target)
            loss = cross_entropy_loss + eeg_attn_loss + eye_attn_loss
            loss.backward()
            for name, params in optimization.param_groups:
                print('打印梯度')
                print('-->name:', name, '-->grad_requirs:', params.requires_grad, \
                      ' -->grad_value:', params.grad)
            clip_grad_norm_(net.parameters(), max_norm=10)
            optimization.step()
            running_loss += loss.item()
            # print("batch loss", loss.item())
            _, prediction = torch.max(out.data, dim=-1)
            total += target.size(0)
            correct += prediction.eq(target.data).cpu().sum().item()
        cur_loss = running_loss / len(train_loader)
        cur_acc = correct / total
        # print(cur_acc, correct, total)
        if isinstance(cur_acc, torch.Tensor):
            cur_acc = cur_acc.item()
        if isinstance(cur_loss, torch.Tensor):
            cur_loss = cur_loss.item()
        print('Loss: %.10f | Acc: %.3f%% (%d/%d)' % (
            cur_loss, 100 * cur_acc, correct, total))
        running_loss_list.append(cur_loss)
        running_acc_list.append(cur_acc)
        scheduler_warmup.step()
        if epoch % 1 == 0:
            net.eval()
            print("start evaluating...")
            test_loss = 0.0
            test_correct = 0.0
            test_total = 0.0
            for i, (feature, target) in enumerate(val_loader):
                eeg = feature[:, :32]
                peripheral = feature[:, 32:]
                eeg = eeg.reshape(-1, 32, 128)
                eeg = eeg.to(device)
                peripheral = peripheral.reshape(-1, 8, 128)
                peripheral = peripheral.to(device)
                target = target.type(torch.LongTensor).to(device)
                with torch.no_grad():
                    out, eeg_attn, eye_attn = net(eeg, peripheral)
                    loss = criterion(out, target) + criterion_attn(eeg_attn, target) + criterion_attn(eye_attn,
                                                                                  target)
                    test_loss += loss.item()
                    _, prediction = torch.max(out.data, dim=-1)
                    # print(prediction)
                    test_total += target.size(0)
                    test_correct += prediction.eq(target.data).cpu().sum().item()
            test_acc = test_correct / test_total
            test_loss = test_loss / len(test_loader)
            if isinstance(test_acc, torch.Tensor):
                test_acc = test_acc.item()
            if isinstance(test_loss, torch.Tensor):
                val_loss = test_loss.item()
            print('Testset Loss: %.10f | Test-Acc: %.3f%% (%d/%d)' % (
                test_loss, 100 * test_acc, test_correct, test_total))
            testing_acc_list.append(test_acc)
            testing_loss_list.append(test_loss)
            if test_acc > best_acc:
                best_acc = test_acc
                print("better model founded in testsets, start saving new model")
                model_name = '%s' % (net.__class__.__name__)
                state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'current_loss': test_loss
                }
                torch.save(state, os.path.join(save_model_path, model_name))
    # 开始计算测试集
    checkpoint = torch.load(os.path.join(save_model_path, net.__class__.__name__))
    net.load_state_dict(checkpoint['net'])
    print("start evaluating...")
    testing_loss = 0.0
    test_correct = 0.0
    test_total = 0.0
    for i, (feature, target) in enumerate(test_loader):
        eeg = feature[:, :32]
        peripheral = feature[:, 32:]
        eeg = eeg.reshape(-1, 32, 128)
        eeg = eeg.to(device)
        peripheral = peripheral.reshape(-1, 8, 128)
        peripheral = peripheral.to(device)
        target = target.type(torch.LongTensor).to(device)
        with torch.no_grad():
            out, eeg_attn, eye_attn = net(eeg, peripheral)
            loss = criterion(out, target) + criterion_attn(eeg_attn, target) + criterion_attn(eye_attn, target)
            testing_loss += loss.item()
            _, prediction = torch.max(out.data, dim=-1)
            # print(prediction)
            test_total += target.size(0)
            test_correct += prediction.eq(target.data).cpu().sum().item()
    test_acc = test_correct / test_total
    test_loss = testing_loss / len(test_loader)
    if isinstance(test_acc, torch.Tensor):
        test_acc = test_acc.item()
    if isinstance(test_loss, torch.Tensor):
        test_loss = test_loss.item()
    print('Testset Loss: %.10f | Acc: %.3f%% (%d/%d)' % (
        test_loss, 100 * test_acc, test_correct, test_total))
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)
    plot_acc_loss_curve({'train_loss': running_loss_list,
                         'train_acc': running_acc_list,
                         'test_loss': testing_loss_list,
                         'test_acc': testing_acc_list}, net.__class__.__name__, exp_des)
    pd.DataFrame.from_dict({
        'test_loss': test_loss_list,
        'test_acc': test_acc_list
    }).to_csv('./subject_independent_results/deap_individual_%d_%s.csv' % (individual, class_list[class_target]), mode='w', index=False,
              header=True, encoding='utf-8')

def subject_dependent_k_fold(individual=1, class_target=4, k_fold = 5):
    # k-fold cv
    class_list = [
        "Valence",
        "Arousal",
        "Dominance",
        "Liking",
        "Valence-Arousal"
    ]
    class_nums = [2, 2, 2, 2, 4]
    test_loss_list = []  # 记录每一折验证的loss
    test_acc_list = []  # 记录每一折验证的acc
    test_precision_list = [] # 记录每一折验证的precision
    test_recall_list = [] # 记录每一折验证的recall
    test_f1_list = [] #记录每一折验证的f1
    test_accuray_list = [] # 记录每一折验证的准确率
    # prepare data
    nor_method = 0
    label_smooth = 0.3
    shuffle = True

    
    # reading the data in the whole dataset
    deap = DEAP(individual=individual)
    k_fold_data = deap.get_kfold_X_Y2(k_fold)
    for fold, (train_X, train_Y, test_X, test_Y) in enumerate(k_fold_data):
        print("start {} th cross-validation".format(fold))
        train_X, train_Y, test_X, test_Y = deap_normalization(train_X, train_Y, test_X, test_Y, nor_method=1, merge=2,
                                                              column=0)


        # Hyper-parameters
        epochs = 80
        batch_size = 512
        learning_rate = 1e-4
        criterion = LabelSmoothSoftmax(lb_smooth=label_smooth)
        # criterion_attn = CrossEntropyLoss()
        print("starting subject-dependent %d-th CV training experiments on individual %d class %s" % (fold,
            individual, class_list[class_target]))
    
        print("train_X shape", train_X.shape)
        print("train_Y shape", train_Y.shape)
        print("test_X shape", test_X.shape)
        print("test_Y shape", test_Y.shape)
    
        train_Y, test_Y = train_Y[:, class_target].squeeze(), test_Y[:, class_target].squeeze()

        print("{}-th CV\t train X shape {}\n".format(fold, train_X.shape))
        print("{}-th CV\t train Y shape {}\n".format(fold, train_Y.shape))
        print("{}-th CV\t test X shape {}\n".format(fold, test_X.shape))
        print("{}-th CV\t test Y shape {}\n".format(fold, test_Y.shape))
        print("train Y == 0\t{}".format(sum(train_Y == 0)))
        print("train Y == 1\t{}".format(sum(train_Y == 1)))
        print("train Y == 2\t{}".format(sum(train_Y == 2)))
        print("train Y == 3\t{}".format(sum(train_Y == 3)))
        print("test Y == 0\t{}".format(sum(test_Y == 0)))
        print("test Y == 1\t{}".format(sum(test_Y == 1)))
        print("test Y == 2\t{}".format(sum(test_Y == 2)))
        print("test Y == 3\t{}".format(sum(test_Y == 3)))


        train_loader = DataLoader(dataset=DEAP_DATASET(train_X, train_Y), batch_size=batch_size, shuffle=shuffle,
                                  num_workers=0)
        test_loader = DataLoader(dataset=DEAP_DATASET(test_X, test_Y), batch_size=batch_size, shuffle=shuffle,
                                 num_workers=0)
    
        exp_des = "%d_dependent_%dth_cv_%s_%s_%d_%d_%s" % (
            individual, fold,'shuffle' if shuffle else "without_shuffle", 'deap', epochs, batch_size,
            class_list[class_target])
    
        print("model construction...")
        net = Hierarchical_ATTN_With_Senti_Map(class_num=class_nums[class_target])
        # if fine_tuning we continue train the pretrained model
        net = net.to(device)
        save_model_path = '../../saved_models/%s/deap/subject_%d/%s/fold%d' % (
            net.__class__.__name__, individual, class_list[class_target], fold)
    
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        optimization = RMSprop(net.parameters(), lr=learning_rate, weight_decay=0.01)

        running_loss_list = []
        running_acc_list = []
        validate_loss_list = []
        validate_acc_list = []
        best_acc = -1
        print("start training...")
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimization, T_max=epochs)
        scheduler_warmup = GradualWarmupScheduler(optimizer=optimization, multiplier=10,
                                                  total_epoch=np.ceil(0.1 * epochs),
                                                  after_scheduler=scheduler_cosine)
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, (feature, target) in enumerate(train_loader):
                feature = feature.reshape(-1, 40, 5)
                optimization.zero_grad()
                # print("训练集label:{}".format(target))
                # print("脏数据统计", torch.sum(torch.isnan(feature), dim=0))
                eeg = feature[:, :32, :]
                peripheral = feature[:, 32:, :]
                eeg = eeg.reshape(-1, 32, 5)
                peripheral = peripheral.reshape(-1, 8, 5)
                eeg = eeg.to(device)
                peripheral = peripheral.to(device)
                target = target.type(torch.LongTensor).to(device)
                #print(eeg.shape, peripheral.shape)
                out = net(eeg, peripheral)
                # print("训练集", out.data[:5])
                # print("训练集",eeg_attn.shape, eeg_attn.data[:5])
                # print("训练集",eye_attn.shape, eye_attn.data[:5])
                # print("batch output",out[0])
                cross_entropy_loss = criterion(out, target)
                # eeg_attn_loss = criterion_attn(eeg_attn, target)
                # eye_attn_loss = criterion_attn(eye_attn, target)
                # loss = cross_entropy_loss
                # print("交叉熵损失", cross_entropy_loss.data)
                # print("eeg注意力损失", eeg_attn_loss.data)
                # print("eye注意力损失", eeg_attn_loss.data)
                cross_entropy_loss.backward()
                clip_grad_norm_(net.parameters(), max_norm=10)
                # for name, parms in net.named_parameters():
                #     print('打印梯度')
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                #           ' -->grad_value:', parms.grad)
                optimization.step()
                running_loss += cross_entropy_loss.item()
                # print("batch loss", loss.item())
                _, prediction = torch.max(out.data, dim=-1)
                # print('训练集', prediction[:5])
                total += target.size(0)
                correct += prediction.eq(target.data).cpu().sum().item()
            cur_loss = running_loss / total
            cur_acc = correct / total
            # print(cur_acc, correct, total)
            if isinstance(cur_acc, torch.Tensor):
                cur_acc = cur_acc.item()
            if isinstance(cur_loss, torch.Tensor):
                cur_loss = cur_loss.item()
            print('Training Loss: %.10f | Training Acc: %.3f%% (%d/%d)' % (
                cur_loss, 100 * cur_acc, correct, total))
            running_loss_list.append(cur_loss)
            running_acc_list.append(cur_acc)
            scheduler_warmup.step()
            if epoch % 5 == 0:
                net.eval()
                print("start evaluating...")
                validate_loss = 0.0
                validate_correct = 0.0
                validate_total = 0.0
                for i, (feature, target) in enumerate(test_loader):
                    feature = feature.reshape(-1, 40, 5)
                    # print("训练集label:{}".format(target))
                    eeg = feature[:, :32, :]
                    peripheral = feature[:, 32:, :]
                    eeg = eeg.reshape(-1, 32, 5)
                    peripheral = peripheral.reshape(-1, 8, 5)
                    eeg = eeg.to(device)
                    peripheral = peripheral.to(device)
                    target = target.type(torch.LongTensor).to(device)
                    with torch.no_grad():
                        out = net(eeg, peripheral)
                        # print("c集", out.data[:5])
                        # print("c集", eeg_attn.data[:5])
                        # print("c集", eye_attn.data[:5])
                        loss = criterion(out, target)
                        validate_loss += loss.item()
                        _, prediction = torch.max(out.data, dim=-1)
                        # print('验证集',prediction[:10])
                        validate_total += target.size(0)
                        validate_correct += prediction.eq(target.data).cpu().sum().item()
                        # print("验证集相等:{}".format(prediction.eq(target.data)))
                validate_acc = validate_correct / validate_total
                validate_loss = validate_loss / validate_total
                if isinstance(validate_acc, torch.Tensor):
                    validate_acc = validate_acc.item()
                if isinstance(validate_loss, torch.Tensor):
                    validate_loss = validate_loss.item()
                print('Validate Loss: %.10f | Validate-Acc: %.3f%% (%d/%d)' % (
                    validate_loss, 100 * validate_acc, validate_correct, validate_total))
                validate_acc_list.append(validate_acc)
                validate_loss_list.append(validate_loss)
                if validate_acc > best_acc:
                    best_acc = validate_acc
                    print("better model founded in validating sets, start saving new model")
                    model_name = '%s' % (net.__class__.__name__)
                    state = {
                        'net': net.state_dict(),
                        'epoch': epoch,
                        'best_acc': best_acc,
                        'current_loss': validate_loss
                    }
                    torch.save(state, os.path.join(save_model_path, model_name))
        # 开始计算测试集
        checkpoint = torch.load(os.path.join(save_model_path, net.__class__.__name__))
        net.load_state_dict(checkpoint['net'])
        net.eval()
        print("start evaluating...")
        testing_loss = 0.0
        test_correct = 0.0
        test_total = 0.0
        y_pre = []
        y_true = []
        for i, (feature, target) in enumerate(test_loader):
            feature = feature.reshape(-1, 40, 5)
            eeg = feature[:, :32, :]
            peripheral = feature[:, 32:, :]
            eeg = eeg.reshape(-1, 32, 5)
            peripheral = peripheral.reshape(-1, 8, 5)
            eeg = eeg.to(device)
            peripheral = peripheral.to(device)
            target = target.type(torch.LongTensor).to(device)
            y_true.extend(target.cpu().numpy().tolist())
            with torch.no_grad():
                out = net(eeg, peripheral)
                loss = criterion(out, target)
                testing_loss += loss.item()
                _, prediction = torch.max(out.data, dim=-1)
                y_pre.extend(prediction.cpu().numpy().tolist())
                # print(prediction)
                test_total += target.size(0)
                test_correct += prediction.eq(target.data).cpu().sum().item()
        test_acc = test_correct / test_total
        test_loss = testing_loss / test_total
        if isinstance(test_acc, torch.Tensor):
            test_acc = test_acc.item()
        if isinstance(test_loss, torch.Tensor):
            test_loss = test_loss.item()
        print('Test Loss: %.10f | Test Acc: %.3f%% (%d/%d)' % (
            test_loss, 100 * test_acc, test_correct, test_total))

        plot_acc_loss_curve({'train_loss': running_loss_list,
                             'train_acc': running_acc_list,
                             'test_loss': validate_loss_list,
                             'test_acc': validate_acc_list}, net.__class__.__name__, exp_des)
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        test_precision_list.append(precision_score(y_true, y_pre, average='macro'))
        test_recall_list.append(recall_score(y_true, y_pre, average='macro'))
        test_f1_list.append(f1_score(y_true, y_pre, average='macro'))
        test_accuray_list.append(accuracy_score(y_true, y_pre))
        with open('./cv_results/{}_classification_reports.txt'.format(class_list[class_target]), 'a+') as f:
            f.write("*********** {}-th CV Predict results of individual {}***********\n".format(fold, individual))

            f.write(
                "classification reports:\n{}\nconfusion matrix:\n{}\noytx_accuracy_score:{}\noytx_precision_score:{}\noytx_recall_score:{}\noytx_f1_score:{}\n".format(
                    classification_report(y_true, y_pre), confusion_matrix(y_true, y_pre),
                    accuracy_score(y_true, y_pre), precision_score(y_true, y_pre, average='macro'),
                    recall_score(y_true, y_pre, average='macro'), f1_score(y_true, y_pre, average='macro')))

            f.write("******************************************************\n")


    df = pd.DataFrame.from_dict({
        'test_loss': test_loss_list,
        'test_acc': test_acc_list,
        "test_accuracy":test_accuray_list,
        "test_precision":test_precision_list,
        "test_recall":test_recall_list,
        "test_f1":test_f1_list
    })
    df_mean = df.mean()
    df_std = df.std()
    df = df.append(df_mean, ignore_index=True)
    df = df.append(df_std, ignore_index=True)
    df.to_csv('./cv_results/deap_individual_%d_%s.csv' % (individual, class_list[class_target]), mode='w', index=False,
              header=True, encoding='utf-8')


def subject_dependent_k_fold_128(individual=1, class_target=4, k_fold=5):
    # k-fold cv
    class_list = [
        "Valence",
        "Arousal",
        "Dominance",
        "Liking",
        "Valence-Arousal"
    ]
    class_nums = [2, 2, 2, 2, 4]
    test_loss_list = []  # 记录每一折验证的loss
    test_acc_list = []  # 记录每一折验证的acc
    test_precision_list = []  # 记录每一折验证的precision
    test_recall_list = []  # 记录每一折验证的recall
    test_f1_list = []  # 记录每一折验证的f1
    test_accuray_list = []  # 记录每一折验证的准确率
    # prepare data
    nor_method = 0
    label_smooth = 0.1
    shuffle = True

    # reading the data in the whole dataset
    deap = DEAP128(individual=individual)
    k_fold_data = deap.get_kfold_X_Y2(k_fold)
    for fold, (train_X, train_Y, test_X, test_Y) in enumerate(k_fold_data):
        print("start {} th cross-validation".format(fold))
        train_X, train_Y, test_X, test_Y = deap_normalization(train_X, train_Y, test_X, test_Y, nor_method=0, merge=1,
                                                              column=0)

        # Hyper-parameters
        epochs = 150
        batch_size = 512
        learning_rate = 1e-4
        criterion = LabelSmoothSoftmax(lb_smooth=label_smooth)
        # criterion_attn = CrossEntropyLoss()
        print("starting subject-dependent %d-th CV training experiments on individual %d class %s" % (fold,
                                                                                                      individual,
                                                                                                      class_list[
                                                                                                          class_target]))

        print("train_X shape", train_X.shape)
        print("train_Y shape", train_Y.shape)
        print("test_X shape", test_X.shape)
        print("test_Y shape", test_Y.shape)

        train_Y, test_Y = train_Y[:, class_target].squeeze(), test_Y[:, class_target].squeeze()

        print("{}-th CV\t train X shape {}\n".format(fold, train_X.shape))
        print("{}-th CV\t train Y shape {}\n".format(fold, train_Y.shape))
        print("{}-th CV\t test X shape {}\n".format(fold, test_X.shape))
        print("{}-th CV\t test Y shape {}\n".format(fold, test_Y.shape))
        print("train Y == 0\t{}".format(sum(train_Y == 0)))
        print("train Y == 1\t{}".format(sum(train_Y == 1)))
        print("train Y == 2\t{}".format(sum(train_Y == 2)))
        print("train Y == 3\t{}".format(sum(train_Y == 3)))
        print("test Y == 0\t{}".format(sum(test_Y == 0)))
        print("test Y == 1\t{}".format(sum(test_Y == 1)))
        print("test Y == 2\t{}".format(sum(test_Y == 2)))
        print("test Y == 3\t{}".format(sum(test_Y == 3)))

        train_loader = DataLoader(dataset=DEAP_DATASET(train_X, train_Y), batch_size=batch_size, shuffle=shuffle,
                                  num_workers=0)
        test_loader = DataLoader(dataset=DEAP_DATASET(test_X, test_Y), batch_size=batch_size, shuffle=shuffle,
                                 num_workers=0)

        exp_des = "%d_dependent_%dth_cv_%s_%s_%d_%d_%s" % (
            individual, fold, 'shuffle' if shuffle else "without_shuffle", 'deap', epochs, batch_size,
            class_list[class_target])

        print("model construction...")
        net = Hierarchical_ATTN_With_Senti_Map(class_num=class_nums[class_target])
        # if fine_tuning we continue train the pretrained model
        net = net.to(device)
        save_model_path = '../../saved_models/%s/deap/subject_%d/%s/fold%d' % (
            net.__class__.__name__, individual, class_list[class_target], fold)

        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        optimization = RMSprop(net.parameters(), lr=learning_rate, weight_decay=0.01)

        running_loss_list = []
        running_acc_list = []
        validate_loss_list = []
        validate_acc_list = []
        best_acc = -1
        print("start training...")
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimization, T_max=epochs)
        scheduler_warmup = GradualWarmupScheduler(optimizer=optimization, multiplier=10,
                                                  total_epoch=np.ceil(0.1 * epochs),
                                                  after_scheduler=scheduler_cosine)
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, (feature, target) in enumerate(train_loader):
                feature = feature.reshape(-1, 40, 128)
                optimization.zero_grad()
                # print("训练集label:{}".format(target))
                # print("脏数据统计", torch.sum(torch.isnan(feature), dim=0))
                eeg = feature[:, :32, :]
                peripheral = feature[:, 32:, :]
                eeg = eeg.reshape(-1, 32, 128)
                peripheral = peripheral.reshape(-1, 8, 128)
                eeg = eeg.to(device)
                peripheral = peripheral.to(device)
                target = target.type(torch.LongTensor).to(device)
                # print(eeg.shape, peripheral.shape)
                out = net(eeg, peripheral)
                # print("训练集", out.data[:5])
                # print("训练集",eeg_attn.shape, eeg_attn.data[:5])
                # print("训练集",eye_attn.shape, eye_attn.data[:5])
                # print("batch output",out[0])
                cross_entropy_loss = criterion(out, target)
                # eeg_attn_loss = criterion_attn(eeg_attn, target)
                # eye_attn_loss = criterion_attn(eye_attn, target)
                # loss = cross_entropy_loss
                # print("交叉熵损失", cross_entropy_loss.data)
                # print("eeg注意力损失", eeg_attn_loss.data)
                # print("eye注意力损失", eeg_attn_loss.data)
                cross_entropy_loss.backward()
                clip_grad_norm_(net.parameters(), max_norm=10)
                # for name, parms in net.named_parameters():
                #     print('打印梯度')
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                #           ' -->grad_value:', parms.grad)
                optimization.step()
                running_loss += cross_entropy_loss.item()
                # print("batch loss", loss.item())
                _, prediction = torch.max(out.data, dim=-1)
                # print('训练集', prediction[:5])
                total += target.size(0)
                correct += prediction.eq(target.data).cpu().sum().item()
            cur_loss = running_loss / total
            cur_acc = correct / total
            # print(cur_acc, correct, total)
            if isinstance(cur_acc, torch.Tensor):
                cur_acc = cur_acc.item()
            if isinstance(cur_loss, torch.Tensor):
                cur_loss = cur_loss.item()
            print('Training Loss: %.10f | Training Acc: %.3f%% (%d/%d)' % (
                cur_loss, 100 * cur_acc, correct, total))
            running_loss_list.append(cur_loss)
            running_acc_list.append(cur_acc)
            scheduler_warmup.step()
            if epoch % 5 == 0:
                net.eval()
                print("start evaluating...")
                validate_loss = 0.0
                validate_correct = 0.0
                validate_total = 0.0
                for i, (feature, target) in enumerate(test_loader):
                    feature = feature.reshape(-1, 40, 128)
                    # print("训练集label:{}".format(target))
                    eeg = feature[:, :32, :]
                    peripheral = feature[:, 32:, :]
                    eeg = eeg.reshape(-1, 32, 128)
                    peripheral = peripheral.reshape(-1, 8, 128)
                    eeg = eeg.to(device)
                    peripheral = peripheral.to(device)
                    target = target.type(torch.LongTensor).to(device)
                    with torch.no_grad():
                        out = net(eeg, peripheral)
                        # print("c集", out.data[:5])
                        # print("c集", eeg_attn.data[:5])
                        # print("c集", eye_attn.data[:5])
                        loss = criterion(out, target)
                        validate_loss += loss.item()
                        _, prediction = torch.max(out.data, dim=-1)
                        # print('验证集',prediction[:10])
                        validate_total += target.size(0)
                        validate_correct += prediction.eq(target.data).cpu().sum().item()
                        # print("验证集相等:{}".format(prediction.eq(target.data)))
                        print("classification reports {}\n".format(classification_report(target.cpu().numpy(), prediction.cpu().numpy())))
                        print("confusion matrix {}".format(
                            confusion_matrix(target.cpu().numpy(), prediction.cpu().numpy())))
                validate_acc = validate_correct / validate_total
                validate_loss = validate_loss / validate_total
                if isinstance(validate_acc, torch.Tensor):
                    validate_acc = validate_acc.item()
                if isinstance(validate_loss, torch.Tensor):
                    validate_loss = validate_loss.item()
                print('Validate Loss: %.10f | Validate-Acc: %.3f%% (%d/%d)' % (
                    validate_loss, 100 * validate_acc, validate_correct, validate_total))
                validate_acc_list.append(validate_acc)
                validate_loss_list.append(validate_loss)
                if validate_acc > best_acc:
                    best_acc = validate_acc
                    print("better model founded in validating sets, start saving new model")
                    model_name = '%s' % (net.__class__.__name__)
                    state = {
                        'net': net.state_dict(),
                        'epoch': epoch,
                        'best_acc': best_acc,
                        'current_loss': validate_loss
                    }
                    torch.save(state, os.path.join(save_model_path, model_name))
        # 开始计算测试集
        checkpoint = torch.load(os.path.join(save_model_path, net.__class__.__name__))
        net.load_state_dict(checkpoint['net'])
        net.eval()
        print("start evaluating...")
        testing_loss = 0.0
        test_correct = 0.0
        test_total = 0.0
        y_pre = []
        y_true = []
        for i, (feature, target) in enumerate(test_loader):
            feature = feature.reshape(-1, 40, 128)
            eeg = feature[:, :32, :]
            peripheral = feature[:, 32:, :]
            eeg = eeg.reshape(-1, 32, 128)
            peripheral = peripheral.reshape(-1, 8, 128)
            eeg = eeg.to(device)
            peripheral = peripheral.to(device)
            target = target.type(torch.LongTensor).to(device)
            y_true.extend(target.cpu().numpy().tolist())
            with torch.no_grad():
                out = net(eeg, peripheral)
                loss = criterion(out, target)
                testing_loss += loss.item()
                _, prediction = torch.max(out.data, dim=-1)
                y_pre.extend(prediction.cpu().numpy().tolist())
                # print(prediction)
                test_total += target.size(0)
                test_correct += prediction.eq(target.data).cpu().sum().item()
        test_acc = test_correct / test_total
        test_loss = testing_loss / test_total
        if isinstance(test_acc, torch.Tensor):
            test_acc = test_acc.item()
        if isinstance(test_loss, torch.Tensor):
            test_loss = test_loss.item()
        print('Test Loss: %.10f | Test Acc: %.3f%% (%d/%d)' % (
            test_loss, 100 * test_acc, test_correct, test_total))

        plot_acc_loss_curve({'train_loss': running_loss_list,
                             'train_acc': running_acc_list,
                             'test_loss': validate_loss_list,
                             'test_acc': validate_acc_list}, net.__class__.__name__, exp_des)
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        test_precision_list.append(precision_score(y_true, y_pre, average='macro'))
        test_recall_list.append(recall_score(y_true, y_pre, average='macro'))
        test_f1_list.append(f1_score(y_true, y_pre, average='macro'))
        test_accuray_list.append(accuracy_score(y_true, y_pre))
        with open('./cv_results/{}_classification_reports.txt'.format(class_list[class_target]), 'a+') as f:
            f.write("*********** {}-th CV Predict results of individual {}***********\n".format(fold, individual))

            f.write(
                "classification reports:\n{}\nconfusion matrix:\n{}\noytx_accuracy_score:{}\noytx_precision_score:{}\noytx_recall_score:{}\noytx_f1_score:{}\n".format(
                    classification_report(y_true, y_pre), confusion_matrix(y_true, y_pre),
                    accuracy_score(y_true, y_pre), precision_score(y_true, y_pre, average='macro'),
                    recall_score(y_true, y_pre, average='macro'), f1_score(y_true, y_pre, average='macro')))

            f.write("******************************************************\n")

    df = pd.DataFrame.from_dict({
        'test_loss': test_loss_list,
        'test_acc': test_acc_list,
        "test_accuracy": test_accuray_list,
        "test_precision": test_precision_list,
        "test_recall": test_recall_list,
        "test_f1": test_f1_list
    })
    df_mean = df.mean()
    df_std = df.std()
    df = df.append(df_mean, ignore_index=True)
    df = df.append(df_std, ignore_index=True)
    df.to_csv('./cv_results/deap_individual_%d_%s.csv' % (individual, class_list[class_target]), mode='w', index=False,
              header=True, encoding='utf-8')
if __name__ == '__main__':
    # for c in [4, 0, 1, 2, 3]:
    for i in range(1, 33):
        subject_dependent_k_fold(i, class_target=0, k_fold=10)
    print("experiments done...")
    # subject_dependent(11, class_target=4)
    # # subject-independent
    # _x_list = []
    # _y_list = []
    # for i in range(1, 33):
    #     deap = DEAP(individual=i, normalization=1)
    #     X, Y = deap.get_X_Y()
    #     X = X.transpose((0,2,1,3))
    #     X = X.reshape(-1, 128)
    #     Y = Y.reshape(-1, 5)
    #     _x_list.append(X)
    #     _y_list.append(Y)
    # print(X.shape)
    # for c in [4, 0, 1, 2, 3]:
    #     for j in range(1,33):
    #         subject_independent(_x_list, _y_list, individual=j, class_target=c)
    #
    # # main(1, 'subject_dependent')
    # print("experiment done!")