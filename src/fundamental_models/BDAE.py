"""
author: ouyangtianxiong
date:2019/12/15
des: implement the Bimodal Deep Auto encoder for eeg modality and eye modality
"""
import sys
sys.path.append('../')
from data_set.seed_iv import SEED_IV, SEED_IV_DATASET
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
import os
from sklearn import linear_model, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score,GridSearchCV
from sklearn.preprocessing import Imputer
import numpy as np
import pickle
BATCH_SIZE = 64
class BDAE_SKLearn(nn.Module):
    def __init__(self,
                 visible_units_eeg=310,
                 visible_units_eye=31,
                 hidden_units=100,
                 learning_rate=1e-5,
                 learning_rate_decay=False,
                 use_gpu=False,
        ):
        super(BDAE_SKLearn, self).__init__()
        self.desc = "BDAE which weights and bias are initialized with pretrained sklearn RBM "
        self.visible_units_eeg = visible_units_eeg
        self.visible_units_eye = visible_units_eye
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.use_gpu = use_gpu
        self.batch_size = 32

        # modality -> modality hidden -> shared
        self.fc1 = nn.Linear(in_features=self.visible_units_eeg, out_features=self.hidden_units)
        self.fc2 = nn.Linear(in_features=self.visible_units_eye, out_features=self.hidden_units)
        self.fc3 = nn.Linear(in_features=2 * self.hidden_units, out_features=self.hidden_units)

        # share -> modality hidden -> reconstruct modality visible
        self.fc4 = nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units * 2)
        self.fc5 = nn.Linear(in_features=self.hidden_units, out_features=self.visible_units_eeg)
        self.fc6 = nn.Linear(in_features=self.hidden_units, out_features=self.visible_units_eye)

    def init_weight_bias(self, params):
        assert isinstance(params, tuple), "parameters should be tuple"
        assert len(params) == 6, "parameters list should be 6"
        assert len(params[0]) == 2, "param should contain weights and bias"
        self.fc1.weight.copy_(params[0][0])
        self.fc1.bias.copy_(params[0][1])
        self.fc2.weight.copy_(params[1][0])
        self.fc2.bias.copy_(params[1][1])
        self.fc3.weight.copy_(params[2][0])
        self.fc3.bias.copy_(params[2][1])
        self.fc4.weight.copy_(params[3][0])
        self.fc4.bias.copy_(params[3][1])
        self.fc5.weight.copy_(params[4][0])
        self.fc5.bias.copy_(params[4][1])
        self.fc6.weight.copy_(params[5][0])
        self.fc6.bias.copy_(params[5][1])

    def encoder(self, X):
        eeg_hidden = self.fc1(X[:, :310])
        eye_hidden = self.fc2(X[:, 310:])
        shared = self.fc3(torch.cat((eeg_hidden, eye_hidden),dim=1))
        return shared

    def decoder(self, shared):
        recon = self.fc4(shared)
        eeg_recon = self.fc5(recon[:, :self.hidden_units])
        eye_recon = self.fc6(recon[:, self.hidden_units:])
        return eeg_recon, eye_recon

    def forward(self, input_data):
        "data->shared"
        shared_representation = self.encoder(input_data)
        reconstruct_eeg, reconstruct_eye = self.decoder(shared_representation)
        return reconstruct_eeg, reconstruct_eye

    def fine_tune(self, train_dataloader, num_epochs, batch_size=16, savepath=None):
        """
                Fine tuneing the BDAE using BP method
                :param train_dataloader: input data
                :param num_epochs: train epochs
                :param batch_size: batch_size
                :return:
                """
        lowest_error = float('inf')

        if (isinstance(train_dataloader, torch.utils.data.DataLoader)):
            train_loader = train_dataloader
        else:
            train_loader = torch.utils.data.DataLoader(train_dataloader, batch_size=batch_size)
        print("*" * 100)
        print("starting fine tuning BDAE...")
        optimization = Adam(params=self.parameters())
        entropy_loss = CrossEntropyLoss()
        mse_loss = MSELoss()
        for epoch in range(1, num_epochs + 1):
            n_batches = int(len(train_loader))
            running_loss = 0.0
            running_eeg_loss = 0.0
            running_eye_loss = 0.0
            for i, (X, _) in tqdm(enumerate(train_loader), ascii=True, desc="Fine tuning", file=sys.stdout):
                optimization.zero_grad()
                X = Variable(X)
                reconstruct_eeg, reconstruct_eye = self(X)
                eeg_recon_loss = mse_loss(reconstruct_eeg, X[:, :310])
                eye_recon_loss = mse_loss(reconstruct_eye, X[:, 310:])
                total_loss = eeg_recon_loss + eye_recon_loss
                running_loss += total_loss.data
                running_eeg_loss += eeg_recon_loss.data
                running_eye_loss += eye_recon_loss.data
                total_loss.backward()
                optimization.step()
            print("Epoch:{}\nrunning loss:{}\nrunning eeg reconstruct loss:{}\nrunning eye reconstruct loss:{}".format(
                epoch, running_loss, running_eeg_loss, running_eye_loss
            ))

            # 存储模型参数
            if running_loss < lowest_error:
                lowest_error = running_loss
                print("saving model...")

                torch.save(self, os.path.join(savepath, 'BDAE.pth'))


def rbm_pretrain_weights_bias():
    pass




def evaluation(session, individual):
    restored_path = '../../saved_models/BDAE_egg_eye/%d/%d' % (session, individual)

def main():    # hyper-parameters
    individual = 3
    session = 1
    batch_size = 32
    balance = False
    modal = 'concat'

    # data preparation
    seed_iv = SEED_IV(individual=individual, session=session, modal=modal, balance=balance)
    train_X, train_Y = seed_iv.get_train_data()
    test_X, test_Y = seed_iv.get_test_data()
    X, Y = seed_iv.get_X_Y()

    # Normalization
    train_X = (train_X - train_X.min()) / (train_X.max() - train_X.min())
    test_X = (test_X - test_X.min()) / (test_X.max() - test_X.min())
    X = (X - X.min()) / (X.max() - X.min())

    # construct dataloader
    # seed_iv_train_dataset = SEED_IV_DATASET(X=train_X, Y=train_Y)
    # seed_iv_test_dataset = SEED_IV_DATASET(X=test_X, Y=test_Y)
    seed_iv_dataset = SEED_IV_DATASET(X=X, Y=Y)
    train_dataloader = DataLoader(dataset=seed_iv_dataset,shuffle=True,batch_size=batch_size,num_workers=4)
    # seed_iv_train_loader = DataLoader(dataset=seed_iv_train_dataset, shuffle=False, batch_size=32, num_workers=4)
    # seed_iv_test_loader = DataLoader(dataset=seed_iv_test_dataset, shuffle=True, batch_size=32, num_workers=4)

    # pretrain a rbm modal with sklearn
    logistic = linear_model.LogisticRegression(solver='newton-cg',tol=1)
    rbm_eeg = BernoulliRBM(random_state=0,verbose=True)
    rbm_eye = BernoulliRBM(random_state=0, verbose=True)
    rbm_shared = BernoulliRBM(random_state=0, verbose=True)
    rbm_feature_classifier = Pipeline(
        steps=[('rbm', rbm_eeg)]
    )
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

    # search hyper-parameters
    components = [700,500,450,400,350,300,250,200,150,100,50,20,10]
    learning_rate = [1e-1,1e-2,1e-3,1e-4,1e-5]
    n_epoch = [60,80,100,120]
    rbm_hyperparams = {'rbm__n_components':components,'rbm__learning_rate':learning_rate,'rbm__n_iter':n_epoch}

    print("fine tuning rbm-eeg hyperparameters")
    clf_eeg = GridSearchCV(Pipeline([('rbm',rbm_eeg),('logistic',logistic)]), param_grid=rbm_hyperparams)
    print(X_train[:, :310].shape)
    clf_eeg.fit(X_train[:,:310], y_train)
    print("best parameters set founded on devolopment set:")
    print(str(clf_eeg.best_params_))
    with open('./eeg.pk',mode='wb') as f1:
        pickle.dump(clf_eeg.best_params_, f1)
    f = open('../logs/BDAE_Sklearn.log', mode='a', encoding='utf-8')
    f.write("best parameters set founded on devolopment set:")
    f.write(str(clf_eeg.best_params_))
    print()
    means = clf_eeg.cv_results_['mean_test_score']
    stds = clf_eeg.cv_results_['std_test_score']

    for mean, std ,rbm_hyperparams in zip(means,stds,clf_eeg.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, rbm_hyperparams))
        f.write("%0.3f (+/-%0.03f) for %r\n"
              % (mean, std * 2, rbm_hyperparams))
    f.close()

    # ('imp',Imputer(strategy='mean')),
    print("fine tuning rbm-eye hyperparameters")
    clf_eye = GridSearchCV(Pipeline([('rbm',rbm_eye),('logistic',logistic)]), param_grid=rbm_hyperparams)
    clf_eye.fit(X_train[:,310:],y_train)
    print("best parameters set founded on devolopment set:")
    print(clf_eye.best_params_)
    print()
    f = open('../logs/BDAE_Sklearn.log', mode='a', encoding='utf-8')
    f.write(str(clf_eye.best_params_))
    means = clf_eye.cv_results_['mean_test_score']
    stds = clf_eye.cv_results_['std_test_score']
    for mean, std, rbm_hyperparams in zip(means, stds, clf_eye.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r\n"
              % (mean, std * 2, rbm_hyperparams))
        f.write("%0.3f (+/-%0.03f) for %r\n"
              % (mean, std * 2, rbm_hyperparams))
    f.close()



    # eeg_hidden = clf_eeg.transform(X_train[:,:310])
    # print(eeg_hidden.shape)
    # eye_hidden = clf_eye.transform(X_train[:,310:])
    # print(eye_hidden.shape)
    #
    # hidden = np.hstack((eeg_hidden,eye_hidden))
    # print(hidden.shape)
    # print("fine tuning rbm-shared hyperparameters")
    # clf_shared = GridSearchCV(rbm_eeg, param_grid=rbm_hyperparams)
    # clf_shared.fit(hidden)
    # print("best parameters set founded on devolopment set:")
    # print(clf_shared.best_params_)
    # print()
    # means = clf_shared.cv_results_['mean_test_score']
    # stds = clf_shared.cv_results_['std_test_score']
    # for mean, std, rbm_hyperparams in zip(means, stds, clf_shared.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))








    # savepath = '../../saved_models/BDAE_egg_eye/%d/%d' % (session, individual)
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
    # # modal construction
    # net = BDAE(visible_units_eeg=310, visible_units_eye=31, hidden_units=50, k=2, learning_rate=1e-3,
    #            xavier_init=True)
    # if os.path.exists(os.path.join(savepath, 'BDAE.pth')):
    #     print("checkpoint founded, loading...")
    #     # loading model
    #     net = torch.load(os.path.join(savepath, 'BDAE.pth'))
    #     print("loading checkpoint successful")
    # else:
    #     print("checkpoint hasn't been founded, rebuilding model...")
    #     net = BDAE(visible_units_eeg=310, visible_units_eye=31, hidden_units=50, k=2, learning_rate=1e-3,
    #                xavier_init=True)
    #     net.pre_train_phase(train_dataloader=train_dataloader,num_epochs=50,batch_size=batch_size)
    #     net.fine_tune(train_dataloader=train_dataloader,num_epochs=50,batch_size=batch_size,savepath=savepath)
    #     print("BDAE model rebuilt and training complete...")
    # # compare the discriminative capability of two kind feature
    # from Unimodal_Recognition.svm_classifier import svm_classifier
    # from  sklearn.svm import SVC
    # from sklearn.metrics import accuracy_score
    # # simple fusion concat feature
    # sf_svm = SVC(kernel='linear')
    # # BDAE fusion feature
    # bdae_svm = SVC(kernel='linear')
    # sf_svm.fit(train_X, train_Y)
    # print("1")
    # sf_acc = accuracy_score(test_Y, sf_svm.predict(test_X))
    # print("2")
    # train_X_bdae = torch.Tensor(train_X)
    # test_X_bdae = torch.Tensor(test_X)
    # with torch.no_grad():
    #     Train = net.encoder(train_X_bdae)[0].numpy()
    #     Test = net.encoder(test_X_bdae)[0].numpy()
    # print(train_X[0])
    # print(Train[0])
    # print("5")
    # bdae_svm.fit(Train, train_Y)
    # print("3")
    # bdae_acc = accuracy_score(test_Y,bdae_svm.predict(Test))
    # print("4")
    # print("simple feature fusion:{}\nBDAE feature fusion:{}\n\tdiffer{}".format(
    #     sf_acc, bdae_acc, sf_acc-bdae_acc
    # ))


if __name__=='__main__':
    main()
    # net = BDAE(visible_units_eeg=310,visible_units_eye=31,hidden_units=50,k=2,learning_rate=1e-3,xavier_init=True)
    # print(net.parameters())
    # for params in net.parameters():
    #     print(params)

    net = BDAE_SKLearn(visible_units_eeg=310,visible_units_eye=31,hidden_units=50,learning_rate=1e-3)
    weight1 = torch.zeros([310,50])
    bias1 = torch.zeros(50)
    weight2 = torch.zeros([31,50])
    bias2 = torch.zeros([50])
    weight3 = torch.zeros([100,50])
    bias3 = torch.zeros([50])
    weight4 = torch.zeros([50, 100])
    bias4 = torch.zeros(100)
    weight5 = torch.zeros([50, 310])
    bias5 = torch.zeros([310])
    weight6 = torch.zeros([50, 31])
    bias6 = torch.zeros([31])
    params = ((weight1,bias1),(weight2,bias2),(weight3,bias3),(weight4,bias4),(weight5,bias5),(weight6,bias6))
    net.init_weight_bias(params)
    for param in net.parameters():
        print(param)
