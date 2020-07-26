"""
author: ouyangtianxiong
date: 2019/12/23
des: implements attention-based emotion recognition
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
from torch.optim import Adam,SGD
from torch.nn import CrossEntropyLoss
import numpy as np
from Common_utils.model_evaluation import plot_acc_loss_curve
from Common_utils.model_training import GradualWarmupScheduler, LabelSmoothSoftmax
from Common_utils.basic_module import FCNet
import os
from data_set.seed_iv import SEED_IV, SEED_IV_DATASET

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

        self.v_lin = FCNet(v_size, output_size * 3, drop=drop)
        self.q_lin = FCNet(q_size, output_size * 3, drop=drop)

        self.v_output = FCNet(output_size + v_size, output_size, drop=drop)
        self.q_output = FCNet(output_size + q_size, output_size, drop=drop)

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

            #print("inter head {} \tinterMAF_q2v\n{}".format(i, interMAF_q2v.cpu().detach().numpy()))
            #print("inter head {} \tinterMAF_v2q\n{}".format(i, interMAF_v2q.cpu().detach().numpy()))
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

        self.src_lin = FCNet(src_size, output_size * 2, drop=drop)
        self.tgt_lin = FCNet(tgt_size, output_size, drop=drop)

        self.tgt_output = FCNet(output_size + tgt_size, output_size, drop=drop)

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

        self.v_lin = FCNet(v_size, output_size * 3, drop=drop)
        self.q_lin = FCNet(q_size, output_size * 3, drop=drop)

        self.v_output = FCNet(output_size, output_size, drop=drop)
        self.q_output = FCNet(output_size, output_size, drop=drop)

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
            #print("intra head {} \tinterMAF_q2v\n{}".format(i, dyIntranMAF_v2v.cpu().detach().numpy()))
            #print("intra head {} \tinterMAF_v2q\n{}".format(i, dyIntranMAF_q2q.cpu().detach().numpy()))
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

        self.v_lin = FCNet(v_size, output_size, drop=drop)
        self.q_lin = FCNet(q_size, output_size, drop=drop)

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

        self.v_lin = FCNet(v_size, output_size, drop=drop)
        self.q_lin = FCNet(q_size, output_size, drop=drop)

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
        self.regions = 16  # regions的数量
        self.regions_indexs = [torch.LongTensor(e) for e in
                               [[3, 0, 1, 2, 4], [7, 8, 9, 10, 11], [5, 6], [13, 12], [14, 15, 23, 24, 32, 33],
                                [22, 21, 31, 30, 40, 39], [16, 17, 18, 19, 20], [25, 26, 27, 28, 29],
                                [34, 35, 36, 37, 38], [41, 42], [49, 48], [43, 44, 45, 46, 47],
                                [50, 51, 57], [56, 55, 61], [52, 53, 54], [58, 59, 60]]]
        reginal_extractors = []
        for i in range(self.regions):
            reginal_extractors.append(nn.LSTM(input_size=eeg_size, hidden_size= output_size // 2, batch_first=True, bias=True, bidirectional=True))

        self.reginalFeatureExtractors = nn.ModuleList(reginal_extractors)
        self.bn = nn.BatchNorm1d(num_features=self.regions)


    def forward(self, x):
        """
        :param x: [batch, n_electrode, 5]
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


class EYEFeatureExtractor(nn.Module):
    def __init__(self, eye_size, output_size):
        super(EYEFeatureExtractor, self).__init__()
        self.eye_size = eye_size
        self.output_size = output_size
        self.regions = 5
        self.regions_indexs = [torch.LongTensor(e) for e in
                               [[0,1,2,3,4,5,6,7,8,9,10,11],
                                [12,13,14,15],
                                [16,17],
                                [18,19,20,21],
                                [22,23,24,25,26,27,28,29,30]]]

        eye_extractor = []
        eye_extractor.append(FCNet(in_size=12, out_size=output_size, activate='relu'))
        eye_extractor.append(FCNet(in_size=4, out_size=output_size, activate='relu'))
        eye_extractor.append(FCNet(in_size=2, out_size=output_size, activate='relu'))
        eye_extractor.append(FCNet(in_size=4, out_size=output_size, activate='relu'))
        eye_extractor.append(FCNet(in_size=9, out_size=output_size, activate='relu'))
        self.eyeFeatureExtractor = nn.ModuleList(eye_extractor)
        self.bn = nn.BatchNorm1d(num_features=self.regions)

    def forward(self, x):
        """
        :param x: EYE feature [batch, 31]
        :return: [batch, regons, output_size]
        """
        B = x.shape[0]
        X_regional_output = []
        for i in range(self.regions):
            X_regional_output.append(self.eyeFeatureExtractor[i](x.index_select(dim=1, index=self.regions_indexs[i].to(device))))
        X_regional_feature = torch.cat(X_regional_output, dim=-1).reshape(B, self.regions, self.output_size)
        return self.bn(X_regional_feature)


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        # define number of detector for each sentiment class
        self.lin1 = FCNet(in_features, mid_features, activate='relu', drop=drop)
        self.lin2 = FCNet(mid_features, out_features, drop=drop)
        #
        self.bilinear = nn.Bilinear(in1_features=in_features, in2_features=in_features, out_features=in_features)
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
        out = self.lin1(self.bilinear(v_mean, q_mean))
        #print("classifier out 1", out[0])
        out = self.lin2(out)
        #print("classifier out 2", out[0])
        return out
class Hierarchical_ATTN(nn.Module):
    def __init__(self):
        super(Hierarchical_ATTN, self).__init__()
        self.eye_features = 16 # 256
        self.eeg_features = 16 # 256
        self.hidden_feature = 32 # 256
        self.num_inter_head = 4
        self.num_intra_head = 4
        self.num_block = 1

        assert self.hidden_feature % self.num_inter_head == 0, 'hidden features size can not be divided by header nums, please check!!'
        assert self.hidden_feature % self.num_inter_head == 0, 'hidden features size can not be divided by header nums, please check!!'

        # basic feature extractor
        self.eegFeatureExtractor = EEGFeatureExtractor(eeg_size=5, output_size=self.eeg_features)

        self.eyeFeatureExtractor = EYEFeatureExtractor(eye_size=31, output_size=self.eeg_features)

        # inter- & intra-modality attention flow mechanism for fusion cross modality feature
        self.interIntraBlocks = MultiBlocks(
            num_blocks=self.num_block,
            v_size=self.eeg_features,
            q_size=self.eye_features,
            output_size=self.hidden_feature,
            num_inter_head=self.num_inter_head,
            num_intra_head=self.num_intra_head,
            drop=0.1
        )

        # emotion classifier
        self.classifier = Classifier(
            in_features=self.hidden_feature,
            mid_features=512, out_features=4,
            drop=0.5)
        # self.classifier = Senti_Map_Classifier(
        #     in_features=self.hidden_feature,
        #     mid_features=256, out_features=4,
        #     drop=0.5)
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

def main(session=1, mode='subject_dependent'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # prepare data
    session = session
    balance = False
    shuffle = False
    modal = 'concat'
    nor_method = 1
    label_smooth = 0.1
    fine_tuning = True

    # reading the data in the whole dataset
    all_individual_data = []
    for i in range(1, 16):
        print("contructing dataset...")
        eeg = SEED_IV(session=session, individual=i, modal=modal, shuffle=shuffle, balance=balance,
                      normalization=nor_method)
        _train_X, _train_Y = eeg.get_train_data()
        _test_X, _test_Y = eeg.get_test_data()
        all_individual_data.append([(_train_X, _train_Y), (_test_X, _test_Y)])

    # Hyper-parameters
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epochs = 120
    batch_size = 128
    learning_rate = 1e-3
    criterion = LabelSmoothSoftmax(lb_smooth=label_smooth)
    for idx in range(1, 16):
        if mode == 'subject_dependent':
            train_X, train_Y = all_individual_data[idx-1][0][0], all_individual_data[idx-1][0][1]
            test_X, test_Y = all_individual_data[idx-1][1][0], all_individual_data[idx-1][1][1]
            exp_des = "%d_dependent_in_seesion_%d_%s_%s_%s_%d_%d" % (
                idx, session, 'balance' if balance else 'without_balance',
                'shuffle' if shuffle else "without_shuffle", 'seed', epochs, batch_size)
            print("starting subject-dependent training experiments on individual %d in session %d"% (idx, session))
        elif mode == 'subject_independent':
            train_X = np.vstack([np.vstack((e[0][0], e[1][0])) for i, e in enumerate(all_individual_data) if i != idx-1])
            train_Y = np.hstack([np.hstack((e[0][1], e[1][1])) for i, e in enumerate(all_individual_data) if i != idx-1])
            test_X = np.vstack((all_individual_data[idx-1][0][0], all_individual_data[idx-1][1][0]))
            test_Y = np.hstack((all_individual_data[idx-1][0][1], all_individual_data[idx-1][1][1]))
            exp_des = "%d_independent_as_testset_in_seesion_%d_%s_%s_%s_%d_%d" % (
                idx, session, 'balance' if balance else 'without_balance',
                'shuffle' if shuffle else "without_shuffle", 'seed', epochs, batch_size)
            print("starting subject-independent training experiments with individual %d in session %d as test set" % (idx, session))
        else:
            raise ValueError

        print("train_X shape", train_X.shape)
        print("train_Y shape", train_Y.shape)
        print("test_X shape", test_X.shape)
        print("test_Y shape", test_Y.shape)
        train_loader = DataLoader(dataset=SEED_IV_DATASET(train_X, train_Y), batch_size=batch_size, shuffle=shuffle,
                                  num_workers=4)
        test_loader = DataLoader(dataset=SEED_IV_DATASET(test_X, test_Y), batch_size=batch_size, shuffle=shuffle,
                                 num_workers=4)

        print("model construction...")
        net = Hierarchical_ATTN()
        # if fine_tuning we continue train the pretrained model
        if mode == 'subject_dependent' and fine_tuning:
            load_path = "../../saved_models/%s/session_%d/subject_%d_as_testset" % (net.__class__.__name__, session, idx)
            files = os.listdir(load_path)
            best_model = max(files)
            checkpoint = torch.load(os.path.join(load_path, best_model))
            net.load_state_dict(checkpoint['net'])
            learning_rate = 1e-5
            batch_size = train_X.shape[0]

        net = net.to(device)
        save_model_path = '../../saved_models/%s/session_%d/subject_%d_as_testset' % (
        net.__class__.__name__, session, idx) if mode == 'subject_independent' else '../../saved_models/%s/session_%d/subject_%d' % (
        net.__class__.__name__, session, idx)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        optimization = Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)

        # save model training state
        running_loss_list = []
        running_acc_list = []
        testing_loss_list =[]
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
                #print("脏数据统计", torch.sum(torch.isnan(feature), dim=0))
                eeg = feature[:, :310]
                eye = feature[:, 310:]
                eeg = eeg.reshape(-1, 62, 5)
                eeg = eeg.to(device)
                eye = eye.to(device)
                target = target.type(torch.LongTensor).to(device)
                out = net(eeg, eye)
                #print("batch output",out[0])
                cross_entropy_loss = criterion(out, target)
                cross_entropy_loss.backward()
                clip_grad_norm_(net.parameters(), max_norm=10)
                optimization.step()
                running_loss += cross_entropy_loss.item()
                #print("batch loss", loss.item())
                _, prediction = torch.max(out.data, dim=-1)
                total += target.size(0)
                correct += prediction.eq(target.data).cpu().sum()
            cur_loss = running_loss / len(train_loader)
            cur_acc = correct / total
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
                testing_loss = 0.0
                test_correct = 0.0
                test_total = 0.0
                for i, (feature, target) in enumerate(test_loader):
                    eeg = feature[:, :310]
                    eye = feature[:, 310:]
                    eeg = eeg.reshape(-1, 62, 5)
                    eeg = eeg.to(device)
                    eye = eye.to(device)
                    target = target.type(torch.LongTensor).to(device)
                    with torch.no_grad():
                        out = net(eeg, eye)
                        loss = criterion(out, target)
                        testing_loss += loss.item()
                        _, prediction = torch.max(out.data, dim=-1)
                        # print(prediction)
                        test_total += target.size(0)
                        test_correct += prediction.eq(target.data).cpu().sum()
                test_acc = test_correct / test_total
                test_loss = testing_loss / len(test_loader)
                if isinstance(test_acc, torch.Tensor):
                    test_acc = test_acc.item()
                if isinstance(test_loss, torch.Tensor):
                    test_loss = test_loss.item()
                print('Testset Loss: %.10f | Acc: %.3f%% (%d/%d)' % (
                    test_loss, 100 * test_acc, test_correct, test_total))
                testing_acc_list.append(test_acc)
                testing_loss_list.append(test_loss)
                if test_acc > best_acc:
                    best_acc = test_acc
                    print("better model founded in testsets, start saving new model")
                    model_name = '%s_%s' % (net.__class__.__name__, str(best_acc)[2:6])
                    state = {
                        'net': net.state_dict(),
                        'epoch': epoch,
                        'best_acc': best_acc,
                        'current_loss': test_loss
                    }
                    torch.save(state, os.path.join(save_model_path, model_name))
        plot_acc_loss_curve({'train_loss': running_loss_list,
                            'train_acc': running_acc_list,
                            'test_loss': testing_loss_list,
                            'test_acc': testing_acc_list}, net.__class__.__name__, exp_des)
if __name__ == '__main__':
    for mode in ['subject_independent','subject_dependent']:
        for session in range(1, 4):
            main(session, mode)
    # main(1, 'subject_dependent')
    print("experiment done!")









