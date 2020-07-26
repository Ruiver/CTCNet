"""
author: ouyang tianxiong
date: 2019/10/23
des: implement utilities tool help model training
"""
import sys
sys.path.append('../')
import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from  torch.optim.lr_scheduler import ReduceLROnPlateau

class LabelSmoothSoftmax(nn.Module):
    def __init__(self, lb_smooth=0.1):
        super(LabelSmoothSoftmax, self).__init__()
        self.lb_smooth = lb_smooth
        self.log_softmax = nn.LogSoftmax(dim=1)
        if lb_smooth > 0:
            self.criterion = nn.KLDivLoss(size_average=False)
        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=100000)
        self.confidence = 1.0 - lb_smooth

    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.lb_smooth / (num_tokens -1))
        return one_hot

    def forward(self, logits, label):
        """
        :param logits: tensor of shape(B, n_class)
        :param label:  tensor of shape(N,1)
        :return:
        """
        score = self.log_softmax(logits)
        num_tokens = score.size(1)

        # conduct label_smoothing module
        gtruth = label.view(-1)
        if self.confidence < 1:
            t_data = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)
            if label.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, t_data.unsqueeze(1), self.confidence)
            gtruth = tmp_.detach()
        loss = self.criterion(score, gtruth)
        return loss



TOTAL_BAR_LENGTH = 31
def set_lr(optim, lr):
    """
    reset the learning rate of deep learning model
    :param optim: optimizer
    :param lr: new learning rate
    :return:
    """
    for group in optim.param_groups:
        group['lr'] = lr

def clip_gradient(optim, clip):
    """
    clip gradient
    :param optim: optimizer
    :param clip:
    :return:
    """
    for group in optim.param_groups:
        for param in group["params"]:
            param.grad.data.clamp_(-clip,clip)

def summay_neural_structure(net):
    """
    summary architecture of dl model
    :param net:  model
    :return:
    """
    #print(net)
    params = list(net.parameters())
    total = 0
    for param in params:
        print(param.shape)
        num = 1
        for e in param.shape:
            num *= e
        total += num
    return total

def cal_run_time(start_time):
    """
    computing running time of model
    :param start_time:
    :return:
    """
    ss = int(time.time() - start_time)
    mm = 0
    hh = 0
    dd = 0
    if ss >= 60:
        mm = ss / 60
        ss %= 60
    if mm >= 60:
        hh = mm / 60
        mm %= 60
    if hh >= 24:
        dd = hh / 24
        hh %= 24

    str = ''
    if dd != 0:
        str += '%dD ' % dd
    if hh != 0:
        str += '%dh ' % hh
    if mm != 0:
        str += '%dm ' % mm
    if ss != 0:
        str += '%ds' % ss

    return str
def discard_saved_model(path, limit):
    """
    :param path: path save model
    :param limit: the number of best model want to save
    :return: none
    """
    file = os.listdir(path)
    file_ = [(e, e[-4:]) for e in file]
    file_ = sorted(file_,key=lambda a:a[1], reverse=True)
    if len(file) > limit:
        for f in file_[limit:]:
            if os.path.exists(os.path.join(path, f[0])):
                os.remove(os.path.join(path, f[0]))

class GradualWarmupScheduler(_LRScheduler):
    """
    Gradual warm-up (increasing learning rate in optimizer)
    proposed in "Accurate, Large Minibatch SGD: Training ImageNet in 1 hour"
    Args:
        optimizer (Optimizer): Wrapped Optimizer
        multiplier:target learning rate rate = base lr * multiplier
        total epoch: target learning rate is reached at total_epoch, gradually
        after_schedual: after target_epoch, use this scheduler (eg. ReduceLRonPlateau)
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier<=1.:
            raise ValueError("multiplier should be greater than 1")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch !=0 else 1
        # ReduceLROnPlateau is called at the end of epoch,
        #  whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups,warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics=metrics,epoch=None)
            else:
                self.after_scheduler.step(metrics=metrics, epoch=epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch= epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics=metrics, epoch=epoch)

def main():
    pass
    # net = CNN1(7,0.4,42)
    # optimizer = torch.optim.Adam(params=net.parameters(),lr=0.01)
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,30)
    # scheduler_warmup = GradualWarmupScheduler(optimizer=optimizer,multiplier=8,total_epoch=10,after_scheduler=scheduler_cosine)
    # for epoch in range(40):
    #     scheduler_warmup.step()
    #     print("当前epoch: %d\t当前学习率:\n"%epoch)
    #     print(optimizer.param_groups[0]["lr"])
if __name__=="__main__":
    main()


