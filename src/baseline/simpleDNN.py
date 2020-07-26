"""
author: ouyangtianxiong
date:2019/12/24
des: implementation of simple multiple feed-forward neural network
"""
__author__ = 'ouyangtianxiong'
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam,SGD
from torch.nn import CrossEntropyLoss
from Common_utils.basic_module import FCNet
import os
from data_set.seed_iv import SEED_IV, SEED_IV_DATASET
from torch.utils.data import DataLoader
from Common_utils.model_evaluation import plot_acc_loss_curve



class ResDNN(nn.Module):
    def __init__(self, num_layers,hidden_size=256,drop=0.0):
        super(ResDNN, self).__init__()
        self.num_layers = num_layers
        self.lin = FCNet(in_size = 341, out_size=hidden_size, activate='relu', drop=drop)
        layers = []
        for i in range(num_layers):
            layers.append(FCNet(in_size=hidden_size,out_size=hidden_size,activate='relu', drop=drop))
        self.layers = nn.ModuleList(layers)
        self.classfier = nn.Sequential(
            FCNet(hidden_size, 512, activate='relu', drop=drop),
            FCNet(512, 4, drop=drop)
        )

    def forward(self, x):
        x = self.lin(x)
        x_container = [x]
        for i in range(self.num_layers):
            x1 = self.layers[i](x + x_container[-1])
            x_container.append(x1)
        x = sum(x_container)
        ans = self.classfier(x)
        return ans

class SimpleDNN(nn.Module):
    def __init__(self, num_layers, hidden_size=256, input_size=341, output_size=4,drop=0.0):
        super(SimpleDNN, self).__init__()
        self.num_layers = num_layers
        self.lin = FCNet(in_size=input_size, out_size=hidden_size, activate='relu', drop=drop)
        layers = []
        for i in range(num_layers):
            layers.append(FCNet(in_size=hidden_size,out_size=hidden_size, activate='relu', drop=drop))
        self.layers = nn.ModuleList(layers)
        self.classfier = nn.Sequential(
            FCNet(hidden_size, 512, activate='relu', drop=drop),
            FCNet(512, output_size, drop=drop)
        )

    def forward(self, x):
        x = self.lin(x)
        for i in range(self.num_layers):
            x = self.layers[i](x)
        ans = self.classfier(x)
        return ans

def main(session=1, mode='subject_dependent'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    # prepare data
    session = session
    balance = True
    shuffle = False
    modal = 'concat'
    nor_method = 1

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
    epochs = 50
    batch_size = 64
    learning_rate = 1e-3
    criterion = CrossEntropyLoss()
    for idx in range(1, 16):
        if mode == 'subject_dependent':
            train_X, train_Y = all_individual_data[idx-1][0][0], all_individual_data[idx-1][0][1]
            test_X, test_Y = all_individual_data[idx-1][1][0], all_individual_data[idx-1][1][1]
            exp_des = "%d_dependent_in_seesion_%d_%s_%s_%s_%d_%d" % (
                idx, session, 'balance' if balance else 'without_balance',
                'shuffle' if shuffle else "without_shuffle", 'seed_iv', epochs, batch_size)
            print("starting subject-dependent training experiments on individual %d in session %d"% (idx, session))
        elif mode == 'subject_independent':
            train_X = np.vstack([np.vstack((e[0][0], e[1][0])) for i, e in enumerate(all_individual_data) if i != idx-1])
            train_Y = np.hstack([np.hstack((e[0][1], e[1][1])) for i, e in enumerate(all_individual_data) if i != idx-1])
            test_X = np.vstack((all_individual_data[idx-1][0][0], all_individual_data[idx-1][1][0]))
            test_Y = np.hstack((all_individual_data[idx-1][0][1], all_individual_data[idx-1][1][1]))
            exp_des = "%d_independent_as_testset_in_seesion_%d_%s_%s_%s_%d_%d" % (
                idx, session, 'balance' if balance else 'without_balance',
                'shuffle' if shuffle else "without_shuffle", 'seed_iv', epochs, batch_size)
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
        net = SimpleDNN(num_layers=4, hidden_size=256, drop=0.5)
        net = net.to(device)
        optimization = Adam(net.parameters(), lr=learning_rate)
        save_model_path = '../../saved_models/%s/session_%d/subject_%d_as_testset' % (
            net.__class__.__name__, session,
            idx) if mode == 'subject_independent' else '../../saved_models/%s/session_%d/_subject%d' % (
            net.__class__.__name__, session, idx)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        # save model training state
        running_loss_list = []
        running_acc_list = []
        testing_loss_list =[]
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
                feature = feature.to(device)
                target = target.type(torch.LongTensor).to(device)
                out = net(feature)
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
                    feature = feature.to(device)
                    target = target.type(torch.LongTensor).to(device)
                    with torch.no_grad():
                        out = net(feature)
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
    for mode in ['subject_dependent', 'subject_independent']:
        for session in range(1, 4):
            main(session, mode)
    print("experiment done!")





