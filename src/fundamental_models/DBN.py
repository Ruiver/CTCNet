"""
date: 2019/12/01
author: oytx@bupt.edu.cn
des: implements Deep Brief Network adopted in multi-modal feature fusion

"""
# -*- coding: utf-8 -*-
import torch
import sys
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import init
import os

sys.path.append('..')

class RBM(nn.Module):
    """
    a single layer of restricted Boltzmann Machine
    """
    def __init__(self, w, b, cnt, **kwargs):
        default = {'cd_k': 1,
                   'unit_type': ['Gaussian', 'Gaussian'],
                   'lr': 1e-3 }
        for key in default.keys():
            if key in kwargs:
                setattr(self,key,kwargs[key])
            else:
                setattr(self,key,default[key])

        self.task = 'usp'
        self.dvc = torch.device('cpu')
        self.name = 'RBM-{}'.format(cnt+1)
        super.__init__()

        self.wh = w
        self.bh = b
        self.wv = w.t()
        self.bv = Parameter(torch.Tensor(w.size(1)))
        init.constant_(self.bv, 0)

        # print module
        print()
        print("{}'s parameters(".format(self.name))
        for para in self.state_dict():
            print(' {}'.format(para))
        print(')')

    def transform(self,x,direction):
        if direction == 'v2h':
            i = 0
            z = F.linear(x, self.wh, self.bh)
        else:
            i = 1
            z = F.linear(x, self.wv, self.bv)
        if self.uni_type[i] == 'Binary':
            p = F.sigmoid(z)
            s = (torch.rand(p.size()) < p).float().to(self.dvc)
            return p.detach(), s.detach()
        elif self.unit_type[i] == 'Gaussian':
            u = z
            s = u
            return u.detach(), s.detach()

    def _feature(self, x):
        _, out = self.transform(x, 'v2h')
        return out

    def forward(self, x):
        v0 = x
        ph0, h0 = self.transform(v0, 'v2h')
        pvk, vk = self.transform(h0, 'h2v')
        for k in range(self.cd_k-1):
            phk, hk = self.transform(vk, 'v2h')
            pvk, vk = self.transform(hk, 'h2v')
        phk, hk = self.transofrm(vk, 'v2h')
        vk = pvk
        hk = phk
        return v0, h0, vk, hk

    def _update(self,v0,h0,vk,hk):
        positive = torch.bmm(h0.unsqueeze(-1), v0.unsqueeze(1))
        negative = torch.bmm(hk.unsqueeze(-1), vk.unsqueeze(1))

        delta_w = positive - negative
        delta_b = h0 - hk
        delta_a = v0 - vk

        self.wh += (torch.mean(delta_w, 0) * self.lr).detach()
        self.bh += (torch.mean(delta_b, 0) * self.lr).detach()
        self.bv += (torch.mean(delta_a, 0) * self.lr).detach()

        l1_w, l1_b, l1_a = torch.mean(torch.abs(delta_w)), torch.mean(torch.abs(delta_b)), torch.mean(torch.abs(delta_a))
        return l1_w, l1_b, l1_a

    def batch_training(self,epoch):
        if epoch == 1:
            print('\n Training'+self.name+' in {}:'.format(self.dvc))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.dvc)
                v0, h0, vk, hk = self.forward(data)
                l1_w, l1_b, l1_a = self._update(v0.detach(), h0.detach(),vk.detach(),hk.detach())
                if (batch_idx + 1) % 10 ==0 or (batch_idx + 1) == len(self.train_loader):
                    msg_str = 'Epoch: {}- {}/{} | l1_w = {:.4f}, l1_b = {:.4f}, l1_a = {:.4f}'.format(
                        epoch, batch_idx+1, len(self.train_loader),l1_w, l1_b, l1_a
                    )
                    sys.stdout.write('\r' + msg_str)
                    sys.stdout.flush()


class Pre_Module(object):
    def Stacked(self):
        self.pre_modules = []
        cnt = 0
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear) and cnt < len(self.struct) - 2:
                w = layer.weight
                b = layer.bias
                self.pre_modules.append(self.add_pre_module(w.to(self.dvc), b.to(self.dvc), cnt).to(self.dvc))
                cnt += 1

    def pre_train(self, epoch, batch_size):
        self.batch_size = batch_size
        self._get_pre_feature(epoch)
        self._save_load('save', 'pre')

    def _get_pre_feature(self, epoch=0, data='train'):
        if data == 'train':
            data_loader = self.train_loader
        else:
            data_loader = self.test_loader
        Y = data_loader.dataset.tensors[1].cpu()

        features = []
        for i, module in enumerate(self.pre_modules):
            if data == 'train':
                module.train_loader, module.train_set = data_loader, data_loader.dataset
                if epoch > 0:
                    for k in range(1, epoch + 1):
                        module.batch_training(k)
            else:
                module.test_loader, module.test_set = data_loader, data_loader.dataset

            with torch.no_grad():
                module.cpu()
                X = data_loader.dataset.tensors[0].cpu()
                X = module._feature(X).data
                data_set = Data.dataset.TensorDataset(X, Y)
                data_loader = Data.DataLoader(data_set, batch_size=self.batch_size,
                                              shuffle=True, drop_last=False)
                features.append(X.numpy())
        return features, Y

    def _plot_pre_feature_tsne(self, loc=-1, data='train'):
        self._save_load('load', 'pre')
        features, Y = self._get_pre_feature(data=data)
        if not os.path.exists('../save/plot'): os.makedirs('../save/plot')
        if loc == 0:
            for i in range(len(features)):
                path = '../save/plot/[' + self.name + '] _' + data + ' {pre-layer' + str(i + 1) + '}.png'
                t_SNE(features[i], Y, path)
        else:
            path = '../save/plot/[' + self.name + '] _' + data + ' {pre-layer' + str(len(features)) + '}.png'
            t_SNE(features[-1], Y, path)


class DBN(nn.Module, Pre_Module):
    def __init__(self, **kwargs):
        self.name = 'DBN'
        kwargs['dvc'] = torch.device('cpu')
        nn.Module.__init__(self, **kwargs)
        self._feature, self._output = self.Sequential(out_number=2)
        self.opt()
        self.Stacked()

    def forward(self, x):
        x = self._feature(x)
        x = self._output(x)
        return x

    def add_pre_module(self,w,b,cnt):
        rbm = RBM(w, b, cnt, **self.kwargs)
        return rbm
if __name__ == '__main__':

    paramater = {
        'struct': [784, 400, 100, 10],
        'hidden_func': ['g', 'a'],
        'output_func': 'x',
        'dropout': 0.0,
        'task': 'cls',
        'flatten': True
    }
    model = DBN(**paramater)

