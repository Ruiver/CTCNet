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
import os
from data_set.deap_feature import DEAP, DEAP_DATASET
import pandas as pd
from baseline.simpleDNN import SimpleDNN
device = torch.device('cpu')

def subject_dependent(individual=1, class_target=4):
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
    epochs = 150
    batch_size = 512
    learning_rate = 1e-3
    criterion = LabelSmoothSoftmax(lb_smooth=label_smooth)
    #criterion_attn = CrossEntropyLoss()
    print("starting subject-dependent training experiments on individual %d class %s" % (
        individual, class_list[class_target]))

    print("train_X shape", train_X.shape)
    print("train_Y shape", train_Y.shape)
    print("validate_X shape", validate_X.shape)
    print("validate_Y shape", validate_Y.shape)
    print("test_X shape", test_X.shape)
    print("test_Y shape", test_Y.shape)

    train_Y, test_Y, validate_Y = train_Y[:, class_target].squeeze(), test_Y[:, class_target].squeeze(), validate_Y[:,
                                                                                                         class_target].squeeze()
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
    net = SimpleDNN(num_layers=3, input_size=200, output_size=class_nums[class_target])
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
            feature = feature.reshape(-1, 200)
            optimization.zero_grad()
            # print("脏数据统计", torch.sum(torch.isnan(feature), dim=0))
            feature = feature.to(device)
            target = target.type(torch.LongTensor).to(device)
            out = net(feature)
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
        if epoch % 1 == 0:
            net.eval()
            print("start evaluating...")
            validate_loss = 0.0
            validate_correct = 0.0
            validate_total = 0.0
            for i, (feature, target) in enumerate(validate_loader):
                feature = feature.reshape(-1, 200)
                feature = feature.to(device)
                target = target.type(torch.LongTensor).to(device)
                with torch.no_grad():
                    out = net(feature)
                    # print("c集", out.data[:5])
                    # print("c集", eeg_attn.data[:5])
                    # print("c集", eye_attn.data[:5])
                    loss = criterion(out, target)
                    validate_loss += loss.item()
                    _, prediction = torch.max(out.data, dim=-1)
                    # print('验证集',prediction[:5])
                    validate_total += target.size(0)
                    validate_correct += prediction.eq(target.data).cpu().sum().item()
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
                print("better model founded in testsets, start saving new model")
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
    for i, (feature, target) in enumerate(test_loader):
        feature = feature.reshape(-1, 200)
        feature = feature.to(device)
        target = target.type(torch.LongTensor).to(device)
        with torch.no_grad():
            out = net(feature)
            loss = criterion(out, target)
            testing_loss += loss.item()
            _, prediction = torch.max(out.data, dim=-1)
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
if __name__ == '__main__':
    subject_dependent(1,4)