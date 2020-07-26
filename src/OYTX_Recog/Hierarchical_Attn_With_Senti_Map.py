"""
author: ouyangtianxiong
date: 2019/12/23
des: implements attention-based emotion recognition
Based on code from https://github.com/KaihuaTang/VQA2.0-Recent-Approachs-2018.pytorch
"""
import sys
sys.path.append('../')
__author__ = 'ouyangtianxiong.bupt.edu.cn'
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam,SGD, RMSprop, Adagrad
from torch.nn import CrossEntropyLoss
import numpy as np
from Common_utils.model_evaluation import plot_acc_loss_curve
from Common_utils.model_training import GradualWarmupScheduler, LabelSmoothSoftmax
from Common_utils.basic_module import FCNet
from Common_utils.basic_utils import seed_normalization
import os
from data_set.seed_iv import SEED_IV, SEED_IV_DATASET
from Hierarchical_Attn import MultiBlocks, OneSideInterModalityUpdate, InterModalityUpdate,SingleBlock,EEGFeatureExtractor,EYEFeatureExtractor,Classifier
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
        #print('eeg_activate', eeg_activate.shape, eeg_activate[0])
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
        #('eye_activate', eye_activate.shape, eye_activate[0])
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
        #print('eeg_senti_relevance', eeg_senti_relevance.shape)
        # [batch, emotion_class, r2, k]
        eye_senti_relevance = self.eye_senti_relevance_detect(q)
        #print('eye_senti_relevance', eye_senti_relevance.shape)

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
        #print('attn_eeg', attn_eeg.shape,attn_eeg[0])
        emotion_eeg_attn = attn_eeg.squeeze().sum(dim=2, keepdim=True) / attn_eeg.size(2)
        # [batch, emotion_class] represents the maximum activate in sentiment map
        emotion_eeg_attn = emotion_eeg_attn.squeeze()
        # [batch, emotion_class, r2,1]
        attn_eye = eye_senti_relevance.sum(dim=3, keepdim=True) / self.k
        #print('attn_eye', attn_eye.shape,attn_eye[0])
        emotion_eye_attn = attn_eye.squeeze().sum(dim=2, keepdim=True) / attn_eye.size(2)
        emotion_eye_attn = emotion_eye_attn.squeeze()
        #print('emotion_eeg_attn', emotion_eeg_attn[0])
        #print('emotion_eye_attn', emotion_eye_attn[0])
        # [batch, emotion_class, r1, 1] * [batch, 1, r1, features] = [batch, emotion class, r, features]
        # introduce learnable sentiment relevance
        map_eeg = attn_eeg * eeg.unsqueeze(dim=1)
        map_eye = attn_eye * eye.unsqueeze(dim=1)
        print("attn_eeg {}\n".format(attn_eeg.cpu().detach().numpy()))
        print("attn_eye {}\n".format(attn_eye.cpu().detach().numpy()))

        out = []
        for i in range(self.emotion_class):
            emotion_specific_eeg = map_eeg[:, i, :, :].squeeze().mean(1).squeeze()
            emotion_specific_eye = map_eye[:, i, :, :].squeeze().mean(1).squeeze()
            tmp = self.bilinears[i](emotion_specific_eeg, emotion_specific_eye)
            out1 = self.emotion_classifer[i](tmp)
            out.append(out1)
        out = torch.cat(out, dim=-1)
        return out


class Senti_Map_Classifier2(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Senti_Map_Classifier2, self).__init__()
        # define number of detector for each sentiment class
        self.k = 10
        self.emotion_class = out_features
        self.downconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.emotion_class * self.k, kernel_size=1, stride=1, padding=0,
                      bias=True)
        )

        self.GAP = nn.AvgPool2d(4)

        self.lin1 = FCNet(in_features * self.emotion_class, mid_features, activate='relu', drop=drop)
        self.lin2 = FCNet(mid_features, out_features, drop=drop)

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

        # [batch, emotion_class, r1,1]
        attn_eeg = eeg_senti_relevance.sum(dim=3, keepdim=True) / self.k

        # [batch, emotion_class, r2,1]
        attn_eye = eye_senti_relevance.sum(dim=3, keepdim=True) / self.k

        # [batch, emotion_class, r1, 1] * [batch, 1, r1, features] = [batch, emotion class, r, features]
        # introduce learnable sentiment relevance
        map_eeg = attn_eeg * eeg.unsqueeze(dim=1)
        map_eye = attn_eye * eye.unsqueeze(dim=1)
        # [batch, emotion_class, feature]
        map_eeg = map_eeg.sum(dim=2) / r1
        map_eye = map_eye.sum(dim=2) / r2

        fusion_feature = map_eeg * map_eye
        final = fusion_feature.view(b, -1)
        out = self.lin1(final)
        out = self.lin2(out)
        return out

class Hierarchical_ATTN_With_Senti_Map(nn.Module):
    def __init__(self):
        super(Hierarchical_ATTN_With_Senti_Map, self).__init__()
        self.eye_features = 64 # 256
        self.eeg_features = 64 # 256
        self.hidden_feature = 128 # 256
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
            drop=0.5
        )

        # self.interIntraBlocks = MultiBlocks(
        #     num_blocks=self.num_block,
        #     v_size=self.eeg_features,
        #     q_size=self.eye_features,
        #     output_size=self.hidden_feature,
        #     num_inter_head=self.num_inter_head,
        #     num_intra_head=self.num_intra_head,
        #     drop=0.5
        # )

        # emotion classifier
        # self.classifier = Classifier(
        #     in_features=self.hidden_feature,
        #     mid_features=512, out_features=4,
        #     drop=0.5)
        self.classifier = Senti_Map_Classifier(
            in_features=self.hidden_feature,
            mid_features=512, out_features=4,
            drop=0.5)
    def forward(self, v, q, flag=0):
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
        if flag == 1:
            return v, q
        else:

            # predict logits
            answer = self.classifier(v, q)
            return answer


def test(session=1):
    def build_classifier():
        c_range = np.logspace(-10, 10, 11, base=2)
        gamma_range = np.logspace(-10, 2, 11, base=2)
        kernel = ['rbf']
        parameters = {'C': c_range, "kernel": kernel, "gamma":gamma_range}
        print("building svm classifier")
        svm_clf = SVC()
        gs = GridSearchCV(svm_clf, param_grid=parameters, refit=True, n_jobs=-1, verbose=2)
        return gs

    # prepare data
    balance = True
    shuffle = False
    modal = 'concat'
    svm_clf = build_classifier()
    test_acc_list = []
    test_precision_list = []
    test_recall_list = []
    test_f1_list = []
    for idx in range(1, 16):
        print("contructing dataset...")
        eeg = SEED_IV(session=session, individual=idx, modal=modal, shuffle=shuffle, balance=balance, k_fold=6)
        k_fold_data = eeg.get_training_kfold_data()
        for fold, (train_X, train_Y, test_X, test_Y) in enumerate(k_fold_data):
            train_X, train_Y, test_X, test_Y = seed_normalization(train_X, train_Y, test_X, test_Y, nor_method=1,
                                                                  merge=1,
                                                                  column=0)
            net = Hierarchical_ATTN_With_Senti_Map()
            save_model_path = '../../saved_models/%s/session_%d/subject_%d/fold_%d' % (
                net.__class__.__name__, session, idx, fold // 2)
            net = net.to(device)
            checkpoint = torch.load(os.path.join(save_model_path, net.__class__.__name__))
            net.load_state_dict(checkpoint['net'])
            net.eval()
            with torch.no_grad():
                eeg = train_X[:, :310]
                eye = train_X[:, 310:]
                eeg = eeg.reshape(-1, 62, 5)
                eeg = torch.FloatTensor(eeg).to(device)
                eye = torch.FloatTensor(eye).to(device)
                transformed_eeg, transformed_eye = net(eeg, eye, 1)
                transformed_eeg = transformed_eeg.view(transformed_eeg.size(0), -1).cpu().numpy()
                transformed_eye = transformed_eye.view(transformed_eye.size(0), -1).cpu().numpy()
                new_train_X = np.hstack((transformed_eeg, transformed_eye))
                eeg = test_X[:, :310]
                eye = test_X[:, 310:]
                eeg = eeg.reshape(-1, 62, 5)
                eeg = torch.FloatTensor(eeg).to(device)
                eye = torch.FloatTensor(eye).to(device)
                transformed_eeg, transformed_eye = net(eeg, eye, 1)
                transformed_eeg = transformed_eeg.view(transformed_eeg.size(0), -1).cpu().numpy()
                transformed_eye = transformed_eye.view(transformed_eye.size(0), -1).cpu().numpy()
                new_test_X = np.hstack((transformed_eeg, transformed_eye))
            new_train_X, train_Y, new_test_X, test_Y = seed_normalization(new_train_X, train_Y, new_test_X, test_Y, nor_method=1,
                                                                  merge=0,
                                                                  column=0)
            svm_clf.fit(new_train_X, train_Y)
            predict = svm_clf.predict(new_test_X)
            test_acc_list.append(accuracy_score(test_Y, predict))
            test_precision_list.append(precision_score(test_Y, predict, average='macro'))
            test_recall_list.append(recall_score(test_Y, predict, average="macro"))
            test_f1_list.append(f1_score(test_Y, predict, average="macro"))
            print("classification report\n{}\nACC {}\nconfusion matrix\n{}\n".format(
                classification_report(test_Y, predict, digits=4), accuracy_score(test_Y, predict),
                confusion_matrix(test_Y, predict)))
    df = pd.DataFrame().from_dict({"accuracy": test_acc_list,
                                   "precision": test_precision_list,
                                   "recall": test_recall_list,
                                   "f1": test_f1_list})
    df_mean = df.mean()
    df_std = df.std()
    df = df.append(df_mean, ignore_index=True)
    df = df.append(df_std, ignore_index=True)
    df.to_csv('./session%02d/results4.csv' % (session))

def visualize_matrix(individual=2, session=3, file_path=None):
    balance = True
    shuffle = True
    modal = 'concat'
    eeg = SEED_IV(session=session, individual=individual, modal=modal, shuffle=shuffle, balance=balance, k_fold=3)
    k_fold_data = eeg.get_training_kfold_data()
    for fold, (train_X, train_Y, test_X, test_Y) in enumerate(k_fold_data):
        train_X, train_Y, test_X, test_Y = seed_normalization(train_X, train_Y, test_X, test_Y, nor_method=1, merge=0,
                                                              column=0)
        train_X = train_X.astype(np.float32)
        test_X = test_X.astype(np.float32)
        train_Y = train_Y.astype(np.int32)
        test_Y = test_Y.astype(np.int32)
        net = Hierarchical_ATTN_With_Senti_Map()
        save_model_path = '../../saved_models/%s/session_%d/subject_%d/fold_%d' % (
            net.__class__.__name__, session, individual, fold)
        net = net.to(device)
        checkpoint = torch.load(os.path.join(save_model_path, net.__class__.__name__))
        net.load_state_dict(checkpoint['net'])
        net.eval()
        with torch.no_grad():
            eeg = test_X[:, :310]
            eye = test_X[:, 310:]
            eeg = eeg.reshape(-1, 62, 5)
            for i in range(0,len(eeg), 2):
                b = torch.FloatTensor(eeg[i:i+2]).to(device)
                e = torch.FloatTensor(eye[i:i+2]).to(device)
                print(b.shape)
                print(e.shape)
                out = net(b, e)



def subject_dependent(session=1):
    # prepare data
    session = session
    balance = True
    shuffle = True
    modal = 'concat'
    nor_method = 1
    label_smooth = 0.3
    fine_tuning = True
    best_acc_list = []
    best_precision_list = []
    best_recall_list = []
    best_f1_list = []

    result_save_path = './seed_results/session{}'.format(session)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    # reading the data in the whole dataset
    for idx in range(1, 16):
        best_acc = -1
        best_precision = -1
        best_recall = -1
        best_f1 = -1
        print("contructing dataset...")
        eeg = SEED_IV(session=session, individual=idx, modal=modal, shuffle=shuffle, balance=balance)
        train_X, train_Y, test_X, test_Y = eeg.get_training_data()
        print("train_X shape", train_X.shape)
        print("train_Y shape", train_Y.shape)
        print("test_X shape", test_X.shape)
        print("test_Y shape", test_Y.shape)
        train_X, train_Y, test_X, test_Y = seed_normalization(train_X, train_Y, test_X, test_Y, nor_method=1, merge=0,
                                                              column=0)
        train_X = train_X.astype(np.float32)
        test_X = test_X.astype(np.float32)
        train_Y = train_Y.astype(np.int32)
        test_Y = test_Y.astype(np.int32)
        print("train_X shape", train_X.shape)
        print("train_Y shape", train_Y.shape)
        print("test_X shape", test_X.shape)
        print("test_Y shape", test_Y.shape)
        # Hyper-parameters
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        epochs = 500
        batch_size = 1024
        learning_rate = 1e-4
        criterion = LabelSmoothSoftmax(lb_smooth=label_smooth)

        exp_des = "%d_dependent_in_seesion_%d_%s_%s_%s_%d_%d" % (
            idx, session, 'balance' if balance else 'without_balance',
            'shuffle' if shuffle else "without_shuffle", 'seed', epochs, batch_size)
        print("starting subject-dependent training experiments on individual %d in session %d"% (idx, session))

        print("train_X shape", train_X.shape)
        print("train_Y shape", train_Y.shape)
        print("test_X shape", test_X.shape)
        print("test_Y shape", test_Y.shape)

        print("model construction...")
        net = Hierarchical_ATTN_With_Senti_Map()
        # if fine_tuning we continue train the pretrained model

        net = net.to(device)
        save_model_path =  '../../saved_models/%s/session_%d/subject_%d' % (net.__class__.__name__, session, idx)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        optimization = RMSprop(net.parameters(), lr=learning_rate, weight_decay=0.01)

        # save model training state
        running_loss_list = []
        running_acc_list = []
        testing_loss_list =[]
        testing_acc_list = []
        print("start training...")
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimization, T_max=epochs)
        scheduler_warmup = GradualWarmupScheduler(optimizer=optimization, multiplier=10,
                                                        total_epoch=np.ceil(0.1 * epochs),
                                                        after_scheduler=scheduler_cosine)
        for epoch in range(epochs):
            net.train()
            optimization.zero_grad()
            #print("脏数据统计", torch.sum(torch.isnan(feature), dim=0))
            eeg = train_X[:, :310]
            eye = train_X[:, 310:]
            eeg = eeg.reshape(-1, 62, 5)
            eeg = torch.FloatTensor(eeg).to(device)
            eye = torch.FloatTensor(eye).to(device)
            #print("eeg type {}, eye type {}".format(type(eeg),type(eye)))
            target = torch.LongTensor(train_Y).to(device)
            out = net(eeg, eye)
            #print("batch output",out[0])
            loss = criterion(out, target)
            loss.backward()
            clip_grad_norm_(net.parameters(), max_norm=10)
            optimization.step()
            scheduler_warmup.step()
            running_loss = loss.item()
            #print("batch loss", loss.item())
            _, prediction = torch.max(out.data, dim=-1)
            total = target.size(0)
            correct = prediction.eq(target.data).cpu().sum().item()

            cur_loss = running_loss / len(train_X)
            cur_acc = correct / total
            if isinstance(cur_acc, torch.Tensor):
                cur_acc = cur_acc.item()
            if isinstance(cur_loss, torch.Tensor):
                cur_loss = cur_loss.item()
            print('Epoch %d/%d\tTraining Loss: %.10f | Acc: %.3f%% (%d/%d)' % (epoch, epochs,
                cur_loss, 100 * cur_acc, correct, total))
            running_loss_list.append(cur_loss)
            running_acc_list.append(cur_acc)

            if epoch % 1 == 0:
                net.eval()
                print("start evaluating...")
                eeg = test_X[:, :310]
                eye = test_X[:, 310:]
                eeg = eeg.reshape(-1, 62, 5)
                eeg = torch.FloatTensor(eeg).to(device)
                eye = torch.FloatTensor(eye).to(device)
                target = torch.LongTensor(test_Y).to(device)
                with torch.no_grad():
                    out = net(eeg, eye)
                    loss = criterion(out, target)
                    testing_loss = loss.item()
                    _, prediction = torch.max(out.data, dim=-1)
                    # print(prediction)
                    test_total = target.size(0)
                    test_correct = prediction.eq(target.data).cpu().sum().item()

                    y_pre = prediction.cpu().numpy()
                    y_true = target.cpu().numpy()

                    test_acc = accuracy_score(y_true, y_pre)

                    test_loss = testing_loss / test_total
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
                        best_precision = precision_score(y_true, y_pre, average="macro")
                        best_recall = precision_score(y_true, y_pre, average="macro")
                        best_f1 = f1_score(y_true, y_pre, average="macro")
                        print("better model founded in testsets, start saving new model")
                        model_name = '%s' % (net.__class__.__name__)
                        state = {
                            'net': net.state_dict(),
                            'epoch': epoch,
                            'best_acc': best_acc,
                            'current_loss': test_loss
                        }
                        torch.save(state, os.path.join(save_model_path, model_name))
        best_f1_list.append(best_f1)
        best_acc_list.append(best_acc)
        best_precision_list.append(best_precision)
        best_recall_list.append(best_recall)

        plot_acc_loss_curve({'train_loss': running_loss_list,
                            'train_acc': running_acc_list,
                            'test_loss': testing_loss_list,
                            'test_acc': testing_acc_list}, net.__class__.__name__, exp_des)
    df = pd.DataFrame().from_dict({
        "acc":best_acc_list,
        "precision":best_precision_list,
        "recall":best_recall_list,
        "f1":best_f1_list
    })
    df_mean = df.mean()
    df_std = df.std()
    df = df.append(df_mean, ignore_index=True)
    df = df.append(df_std, ignore_index=True)
    df.to_csv(result_save_path+'/results.csv')


def subject_dependent_CV(session=1):
    # prepare data
    session = session
    balance = True
    shuffle = False
    modal = 'concat'
    nor_method = 1
    label_smooth = 0.3
    fine_tuning = True
    best_acc_list = []
    best_precision_list = []
    best_recall_list = []
    best_f1_list = []

    result_save_path = './seed_cv_results/session{}'.format(session)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    # reading the data in the whole dataset
    for idx in range(1, 16):
        print("contructing dataset...")
        eeg = SEED_IV(session=session, individual=idx, modal=modal, shuffle=shuffle, balance=balance, k_fold=3)
        k_fold_data = eeg.get_training_kfold_data()
        for fold, (train_X, train_Y, test_X, test_Y) in enumerate(k_fold_data):
            best_acc = -1
            best_precision = -1
            best_recall = -1
            best_f1 = -1
            print("train_X shape", train_X.shape)
            print("train_Y shape", train_Y.shape)
            print("test_X shape", test_X.shape)
            print("test_Y shape", test_Y.shape)
            train_X, train_Y, test_X, test_Y = seed_normalization(train_X, train_Y, test_X, test_Y, nor_method=1, merge=0,
                                                                  column=0)
            train_X = train_X.astype(np.float32)
            test_X = test_X.astype(np.float32)
            train_Y = train_Y.astype(np.int32)
            test_Y = test_Y.astype(np.int32)
            # Hyper-parameters
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            epochs = 500
            batch_size = 1024
            learning_rate = 1e-4
            criterion = LabelSmoothSoftmax(lb_smooth=label_smooth)

            exp_des = "%d_dependent_in_seesion_%d_fold%d_%s_%s_%s_%d_%d" % (
                idx, session, fold, 'balance' if balance else 'without_balance',
                'shuffle' if shuffle else "without_shuffle", 'seed', epochs, batch_size)
            print("starting subject-dependent training experiments on individual %d in session %d"% (idx, session))

            print("train_X shape", train_X.shape)
            print("train_Y shape", train_Y.shape)
            print("test_X shape", test_X.shape)
            print("test_Y shape", test_Y.shape)

            print("model construction...")
            net = Hierarchical_ATTN_With_Senti_Map()
            # if fine_tuning we continue train the pretrained model

            net = net.to(device)
            save_model_path =  '../../saved_models/%s/session_%d/subject_%d/fold_%d' % (net.__class__.__name__, session, idx,fold)
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            optimization = RMSprop(net.parameters(), lr=learning_rate, weight_decay=0.01)

            # save model training state
            running_loss_list = []
            running_acc_list = []
            testing_loss_list =[]
            testing_acc_list = []
            print("start training...")
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimization, T_max=epochs)
            scheduler_warmup = GradualWarmupScheduler(optimizer=optimization, multiplier=10,
                                                            total_epoch=np.ceil(0.1 * epochs),
                                                            after_scheduler=scheduler_cosine)
            for epoch in range(epochs):
                net.train()
                optimization.zero_grad()
                #print("脏数据统计", torch.sum(torch.isnan(feature), dim=0))
                eeg = train_X[:, :310]
                eye = train_X[:, 310:]
                eeg = eeg.reshape(-1, 62, 5)
                eeg = torch.FloatTensor(eeg).to(device)
                eye = torch.FloatTensor(eye).to(device)
                #print("eeg type {}, eye type {}".format(type(eeg),type(eye)))
                target = torch.LongTensor(train_Y).to(device)
                out = net(eeg, eye)
                #print("batch output",out[0])
                loss = criterion(out, target)
                loss.backward()
                clip_grad_norm_(net.parameters(), max_norm=10)
                optimization.step()
                scheduler_warmup.step()
                running_loss = loss.item()
                #print("batch loss", loss.item())
                _, prediction = torch.max(out.data, dim=-1)
                total = target.size(0)
                correct = prediction.eq(target.data).cpu().sum().item()

                cur_loss = running_loss / len(train_X)
                cur_acc = correct / total
                if isinstance(cur_acc, torch.Tensor):
                    cur_acc = cur_acc.item()
                if isinstance(cur_loss, torch.Tensor):
                    cur_loss = cur_loss.item()
                print('Epoch %d/%d\tTraining Loss: %.10f | Acc: %.3f%% (%d/%d)' % (epoch, epochs,
                    cur_loss, 100 * cur_acc, correct, total))
                running_loss_list.append(cur_loss)
                running_acc_list.append(cur_acc)

                if epoch % 1 == 0:
                    net.eval()
                    print("start evaluating...")
                    eeg = test_X[:, :310]
                    eye = test_X[:, 310:]
                    eeg = eeg.reshape(-1, 62, 5)
                    eeg = torch.FloatTensor(eeg).to(device)
                    eye = torch.FloatTensor(eye).to(device)
                    target = torch.LongTensor(test_Y).to(device)
                    with torch.no_grad():
                        out = net(eeg, eye)
                        loss = criterion(out, target)
                        testing_loss = loss.item()
                        _, prediction = torch.max(out.data, dim=-1)
                        # print(prediction)
                        test_total = target.size(0)
                        test_correct = prediction.eq(target.data).cpu().sum().item()

                        y_pre = prediction.cpu().numpy()
                        y_true = target.cpu().numpy()

                        test_acc = accuracy_score(y_true, y_pre)

                        test_loss = testing_loss / test_total
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
                            best_precision = precision_score(y_true, y_pre, average="macro")
                            best_recall = precision_score(y_true, y_pre, average="macro")
                            best_f1 = f1_score(y_true, y_pre, average="macro")
                            print("better model founded in testsets, start saving new model")
                            model_name = '%s' % (net.__class__.__name__)
                            state = {
                                'net': net.state_dict(),
                                'epoch': epoch,
                                'best_acc': best_acc,
                                'current_loss': test_loss
                            }
                            torch.save(state, os.path.join(save_model_path, model_name))
            best_f1_list.append(best_f1)
            best_acc_list.append(best_acc)
            best_precision_list.append(best_precision)
            best_recall_list.append(best_recall)

            plot_acc_loss_curve({'train_loss': running_loss_list,
                                'train_acc': running_acc_list,
                                'test_loss': testing_loss_list,
                                'test_acc': testing_acc_list}, net.__class__.__name__, exp_des)
    df = pd.DataFrame().from_dict({
        "acc":best_acc_list,
        "precision":best_precision_list,
        "recall":best_recall_list,
        "f1":best_f1_list
    })
    df_mean = df.mean()
    df_std = df.std()
    df = df.append(df_mean, ignore_index=True)
    df = df.append(df_std, ignore_index=True)
    df.to_csv(result_save_path+'/results.csv')
if __name__ == '__main__':
    # for mode in ['subject_dependent', 'subject_independent']:
    #     for session in range(1, 4):
    #         main(session, mode)
    # main(1, 'subject_dependent')
    visualize_matrix(2, 3)
    # for session in [3,2, 1]:
    #         subject_dependent_CV(session)
    # for session in [3, 2, 1]:
    #     test(session)
    print("experiment done!")