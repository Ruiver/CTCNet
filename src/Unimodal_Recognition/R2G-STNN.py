"""
author: ouyangtianxiong
date:2019/12/16
des: implememnt R2G-STNN according to paper
《From Regional to Global Brain :A novel Hierarchical Spatial-Temporal Neural
Network Model for EEG Emotion Recognition 》2019 TAC

Li, Y., Zheng, W., Wang, L., Zong, Y., & Cui, Z. (2019).
 From Regional to Global Brain: A Novel Hierarchical Spatial-Temporal Neural Network Model for EEG Emotion Recognition. IEEE Transactions on Affective Computing.
"""
__Author__ = "ouyangtianxiong"
import sys
sys.path.append('../')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
from torch.nn import CrossEntropyLoss
import os
import numpy as np
from tqdm import tqdm
from data_set.seed import SEED_SEG,SEED_DATASET
from Common_utils.model_evaluation import plot_acc_loss_curve
import time



class R2G_STNN(nn.Module):

    def __init__(self):
        super(R2G_STNN, self).__init__()
        self.d_r = 100 # region BiLstm的隐层结点数
        self.d_g = 150
        self.d_rt = 200
        self.d_gt = 250
        self.K = 3
        self.regions = 16 # regions的数量
        self.T = 9
        self.n_class = 3
        self.regions_indexs = [torch.LongTensor(e) for e in [[3,0,1,2,4],[7,8,9,10,11],[5,6],[13,12],[14,15,23,24,32,33],
                                                [22,21,31,30,40,39],[16,17,18,19,20],[25,26,27,28,29],
                                                [34,35,36,37,38],[41,42],[49,48],[43,44,45,46,47],
                                                [50,51,57],[56,55,61],[52,53,54],[58,59,60]]]
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.6)
        # 第一层区域特征提取器，每个区域由BiLSTM提取区域特征。
        # 第i个LSTM的输入为(b,d,ni)ni表示该区域的电极数量
        self.reginal_feature_bilstms = nn.ModuleList([nn.LSTM(input_size=5, hidden_size=self.d_r,batch_first=True,bidirectional=True) for _ in range(self.regions)])

        #  区域注意力层
        self.regional_attn_P = nn.Linear(in_features=self.d_r * 2, out_features=self.regions, bias=True)
        #self.regional_attn_Q = nn.Linear(in_features=0, out_features=0, bias=False)

        # 全局特征学习层 (16, 2 * d_r) -> (16, 2*self.d_g)
        self.global_feature_bilstm = nn.LSTM(input_size=2 * self.d_r, hidden_size=self.d_g,bias=True,batch_first=True,bidirectional=True)

        # 全局特征融合层
        self.global_feature_fusion = nn.Linear(in_features=self.regions, out_features=self.K)

        # 区域时间维度的bilstm层 (16*T*200)
        self.reginal_temporal_bilstms = nn.ModuleList([nn.LSTM(input_size=self.d_r * 2,hidden_size=self.d_rt,batch_first=True,bidirectional=True) for _ in range(self.regions)])

        # 全局时间维度的bilstm层
        self.global_temporal_bilstm = nn.LSTM(input_size=self.K * self.d_g *2, hidden_size=self.d_gt,batch_first=True,bias=True,bidirectional=True)

        # 情感分类层
        self.emotion_classfier = nn.Linear(in_features=self.d_rt*2*self.regions + self.d_gt *2, out_features=self.n_class)

        # domain classifier
        self.domain_classifier = nn.Linear(in_features=self.d_rt*2*self.regions + self.d_gt *2, out_features=2)

    def forward(self, X, flag='emotion'):
        # 输入X为(B,d,n,T) -> (B,T,n,d)
        #X = X.permute((0,3,2,1))
        B,T,N,D = X.shape
        # divide to regions
        X_regions_input = [] # 列表存储不同区域的张量输入
        for i in range(self.regions):
            X_regions_input.append(X.index_select(dim=2,index=self.regions_indexs[i].cuda()))

        # 1 compute regional features
        X_regional_lstm_out = []
        for i in range(self.regions):
            shape = X_regions_input[i].shape
            #print(shape)
            # 先转成（B*T,n_i,d）再进LSTM
            hidden_units, _ = self.reginal_feature_bilstms[i](X_regions_input[i].reshape((-1,shape[-2],shape[-1])))
            X_regional_lstm_out.append(hidden_units[:,-1,:].squeeze())
        # X_regional_feature : 列表：元素为tensor [ B*T, regions_num, 2*self.d_r]
        # reshape成(B*T, regions, 2*self.d_r)
        # (B * T, regions, 2* self.d_r)
        X_regional_feature = torch.cat(X_regional_lstm_out, dim=-1).reshape(B*T, -1, 2*self.d_r)
        # (B * T, regions, regions)
        regional_attn_weights = self.softmax(self.regional_attn_P(X_regional_feature))
        # (B * T, regions, 2*self.d_r)
        X_regional_feature_attn = torch.bmm(regional_attn_weights, X_regional_feature)

        # global feature learning
        # (B*T, regions, self.d_g)
        global_feature, _ = self.global_feature_bilstm(X_regional_feature_attn)
        global_ = self.dropout(self.global_feature_fusion(global_feature.permute(0, 2, 1)))

        # (B * T, K, self.d_g)

        global_ = global_.permute(0,2,1)

        regional_temporal_feature = []
        for i in range(self.regions):
            x = X_regional_lstm_out[i].reshape(B,T,-1)
            # temp (B, T, 2*self.d_rt)
            temp,_ = self.reginal_temporal_bilstms[i](x)
            #print(temp.shape)
            regional_temporal_feature.append(temp[:,-1,:].squeeze())
        #(B, T, self.d_rt * 2)
        final_reginal = torch.cat(regional_temporal_feature,dim=-1)
        #print(final_reginal.shape)
        final_reginal = final_reginal.reshape(B,self.regions,-1)
        global_flat = global_.reshape(B,T, -1)
        out,_ = self.global_temporal_bilstm(global_flat)
        #(B, self.d_gt * 2)
        final_global = out[:,-1,:].reshape(B,-1)

        final_reginal = final_reginal.reshape(B, -1)

        final_feature = self.dropout(torch.cat((final_reginal, final_global), dim=-1))

        emotion_pred = self.emotion_classfier(final_feature)

        domain_pred = self.domain_classifier(final_feature)
        
        return emotion_pred if flag == 'emotion' else domain_pred



def main(session,mode='subject_dependent'):
    """
    :param session: session
    :param mode: subject dependent or subject independent
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # prepare data
    session = session
    balance = True
    shuffle = False
    modal = 'concat'
    T = 9
    nor_method = 1

    # reading the data in the whole dataset
    all_individual_data = []
    for i in range(1, 16):
        print("contructing dataset...")
        eeg = SEED_SEG(T_len=T, individual=i, session=session, balance=balance, shuffle=shuffle,
                       normalization=nor_method)
        _train_X, _train_Y = eeg.get_train_data()
        _test_X, _test_Y = eeg.get_test_data()
        all_individual_data.append([(_train_X, _train_Y), (_test_X, _test_Y)])

    # Hyper-parameters
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epochs = 50
    batch_size = 32
    learning_rate = 1e-3
    criterion = CrossEntropyLoss()
    for idx in range(1, 16):
        if mode == 'subject_dependent':
            train_X, train_Y = all_individual_data[idx - 1][0][0], all_individual_data[idx - 1][0][1]
            test_X, test_Y = all_individual_data[idx - 1][1][0], all_individual_data[idx - 1][1][1]
            exp_des = "%d_dependent_in_seesion_%d_%s_%s_%s_%d_%d" % (
                idx, session, 'balance' if balance else 'without_balance',
                'shuffle' if shuffle else "without_shuffle", 'seed', epochs, batch_size)
            print("starting subject-dependent training experiments on individual %d in session %d" % (idx, session))
        elif mode == 'subject_independent':
            train_X = np.vstack(
                [np.vstack((e[0][0], e[1][0])) for i, e in enumerate(all_individual_data) if i != idx - 1])
            train_Y = np.hstack(
                [np.hstack((e[0][1], e[1][1])) for i, e in enumerate(all_individual_data) if i != idx - 1])
            test_X = np.vstack((all_individual_data[idx - 1][0][0], all_individual_data[idx - 1][1][0]))
            test_Y = np.hstack((all_individual_data[idx - 1][0][1], all_individual_data[idx - 1][1][1]))
            exp_des = "%d_independent_as_testset_in_seesion_%d_%s_%s_%s_%d_%d" % (
                idx, session, 'balance' if balance else 'without_balance',
                'shuffle' if shuffle else "without_shuffle", 'seed', epochs, batch_size)
            print("starting subject-independent training experiments with individual %d in session %d as test set" % (
            idx, session))
        else:
            raise ValueError

        print("train_X shape", train_X.shape)
        print("train_Y shape", train_Y.shape)
        print("test_X shape", test_X.shape)
        print("test_Y shape", test_Y.shape)
        train_loader = DataLoader(dataset=SEED_DATASET(train_X, train_Y), batch_size=batch_size, shuffle=shuffle,
                                  num_workers=4)
        test_loader = DataLoader(dataset=SEED_DATASET(test_X, test_Y), batch_size=batch_size, shuffle=shuffle,
                                 num_workers=4)

        print("model construction...")
        net = R2G_STNN()
        net = net.to(device)
        save_model_path = '../../saved_models/%s/%d/%d_as_testset' % (
            net.__class__.__name__, session,
            idx) if mode == 'subject_independent' else '../../saved_models/%s/%d/%d' % (
            net.__class__.__name__, session, idx)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        optimization = Adam(net.parameters(), lr=learning_rate)

        # save model training state
        running_loss_list = []
        running_acc_list = []
        testing_loss_list = []
        testing_acc_list = []
        best_acc = -1
        print("start training...")
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, (feature, target) in enumerate(train_loader):
                optimization.zero_grad()
                shape = feature.shape
                assert len(shape) == 2, "input shape mistake, please check! %d" % len(shape)
                assert len(target.shape) == 1, "target shape mistake, please check! %d" % len(target.shape)
                eeg = feature.reshape(shape[0], T, 62, -1)
                eeg = eeg.to(device)
                target = target.type(torch.LongTensor).to(device)
                out = net(eeg)
                loss = criterion(out, target)
                loss.backward()
                optimization.step()
                running_loss += loss.item()
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

            if epoch % 1 == 0:
                net.eval()
                print("start evaluating...")
                testing_loss = 0.0
                test_correct = 0.0
                test_total = 0.0
                for i, (feature, target) in enumerate(test_loader):
                    optimization.zero_grad()
                    shape = feature.shape
                    assert len(shape) == 2, "input shape mistake, please check! %d" % len(shape)
                    assert len(target.shape) == 1, "target shape mistake, please check! %d" % len(target.shape)
                    eeg = feature.reshape(shape[0], T, 62, -1)
                    eeg = eeg.to(device)
                    target = target.type(torch.LongTensor).to(device)
                    with torch.no_grad():
                        out = net(eeg)
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
                    model_name = 'Hierarchical_Attn_%s' % (str(best_acc)[2:])
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
    for mode in ['subject_dependent','subject_independent']:
        for session in range(1,4):
            main(session, mode)







