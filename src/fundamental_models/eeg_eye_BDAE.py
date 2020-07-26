"""
author: ouyangtianxiong
date:2019/12/11
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
import torch.nn.functional as F
import math
from tqdm import tqdm
import os


BATCH_SIZE = 64
class BDAE(nn.Module):
    def __init__(self,
                 visible_units_eeg=310,
                 visible_units_eye=31,
                 hidden_units=100,
                 k=2,
                 learning_rate=1e-5,
                 learning_rate_decay=False,
                 xavier_init=False,
                 increase_to_cd_k=False,
                 use_gpu=False):
        super(BDAE, self).__init__()
        self.desc = "BDAE"
        self.visible_units_eeg = visible_units_eeg
        self.visible_units_eye = visible_units_eye
        self.hidden_units = hidden_units
        self.k = k
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.xavier_init = xavier_init
        self.increase_to_cd_k = increase_to_cd_k
        self.use_gpu = use_gpu
        self.batch_size = 32

        # Initializing
        if not self.xavier_init:
            # eeg visible to hidden
            self.W1 = torch.randn(self.visible_units_eeg, self.hidden_units) * 0.01
            self.W2 = torch.randn(self.visible_units_eye, self.hidden_units) * 0.01
            self.W3 = torch.randn(2 * self.hidden_units, hidden_units) * 0.01
        else:
            xavire_value_w1 = torch.sqrt(torch.FloatTensor([1.0 / (self.visible_units_eeg + self.hidden_units)]))
            xavire_value_w2 = torch.sqrt(torch.FloatTensor([1.0 / (self.visible_units_eye+self.hidden_units)]))
            xavire_value_w3 = torch.sqrt(torch.FloatTensor([1.0 / (2 * self.hidden_units+self.hidden_units)]))
            self.W1 = - xavire_value_w1 + torch.randn(self.visible_units_eeg, self.hidden_units) * (2 * xavire_value_w1)
            self.W2 = - xavire_value_w2 + torch.randn(self.visible_units_eye, self.hidden_units) * (2 * xavire_value_w2)
            self.W3 = - xavire_value_w2 + torch.randn(2 * self.hidden_units, self.hidden_units) * (2 * xavire_value_w3)

        self.W1_T = torch.Tensor(()).new_zeros((self.hidden_units, self.visible_units_eeg))
        self.W2_T = torch.Tensor(()).new_zeros((self.hidden_units, self.visible_units_eye))
        self.W3_T = torch.Tensor(()).new_zeros((self.hidden_units, 2 * self.hidden_units))

        self.v_eeg_bias_T = torch.zeros((self.visible_units_eeg))
        self.v_eye_bias_T = torch.zeros(self.visible_units_eye)
        self.h_eeg_bias_T = torch.zeros(self.hidden_units)
        self.h_eye_bias_T = torch.zeros(self.hidden_units)

        self.v_eeg_bias = torch.zeros(self.visible_units_eeg)
        self.v_eye_bias = torch.zeros(self.visible_units_eye)
        self.h_eeg_bias = torch.zeros(self.hidden_units)
        self.h_eye_bias = torch.zeros(self.hidden_units)
        self.shared_feature_bias = torch.zeros(self.hidden_units)

    def v2h_eeg(self,X):
        '''
                    convert eeg visible data to its hidden layer
                    :param self:
                    :param X: torch tensor shape = (n_samples, n_features)
                    :return: - X_prob - new hidden layer (probabilities)
                            sample_X_prob - Gibbs sampling of hidden (0 or 1) based on the activation values
                    '''
        X_prob = torch.matmul(X, self.W1)
        X_prob = torch.add(X_prob, self.h_eeg_bias)
        X_prob = torch.sigmoid(X_prob)
        sample_X_prob = self.sampling(X_prob)
        return X_prob, sample_X_prob

    def v2h_eye(self,X):
        '''
                            convert eeg visible data to its hidden layer
                            :param self:
                            :param X: torch tensor shape = (n_samples, n_features)
                            :return: - X_prob - new hidden layer (probabilities)
                                    sample_X_prob - Gibbs sampling of hidden (0 or 1) based on the activation values
                            '''
        X_prob = torch.matmul(X, self.W2)
        X_prob = torch.add(X_prob, self.h_eye_bias)
        X_prob = torch.sigmoid(X_prob)
        sample_X_prob = self.sampling(X_prob)
        return X_prob, sample_X_prob

    def v2h_shared(self, X):
        '''
                                    convert eeg visible data to its hidden layer
                                    :param self:
                                    :param X: torch tensor shape = (n_samples, n_features)
                                    :return: - X_prob - new hidden layer (probabilities)
                                            sample_X_prob - Gibbs sampling of hidden (0 or 1) based on the activation values
                                    '''
        X_prob = torch.matmul(X, self.W3)
        X_prob = torch.add(X_prob, self.shared_feature_bias)
        X_prob = torch.sigmoid(X_prob)
        sample_X_prob = self.sampling(X_prob)
        return X_prob, sample_X_prob

    def h2v_eeg(self,X):
        '''
        reconstruct data from hidden layer
        also does sampling
        :param X: X here is the probabilities in hidden layer
        :return: X_prob the new reconstructed layer (probabilities)
                sample_X_prob - sample of new layer (Gibbs sampling)
        '''
        X_prob = torch.matmul(X, self.W1.transpose(0,1))
        X_prob = torch.add(X_prob, self.v_eeg_bias)
        X_prob = torch.sigmoid(X_prob)
        sample_X_prob = self.sampling(X_prob)
        return X_prob, sample_X_prob

    def h2v_eye(self,X):
        '''
        reconstruct data from hidden layer
        also does sampling
        :param X: X here is the probabilities in hidden layer
        :return: X_prob the new reconstructed layer (probabilities)
                sample_X_prob - sample of new layer (Gibbs sampling)
        '''
        X_prob = torch.matmul(X, self.W2.transpose(0,1))
        X_prob = torch.add(X_prob, self.v_eye_bias)
        X_prob = torch.sigmoid(X_prob)
        sample_X_prob = self.sampling(X_prob)
        return X_prob, sample_X_prob

    def h2v_shared(self,X):
        '''
        reconstruct data from hidden layer
        also does sampling
        :param X: X here is the probabilities in hidden layer
        :return: X_prob the new reconstructed layer (probabilities)
                sample_X_prob - sample of new layer (Gibbs sampling)
        '''
        X_prob = torch.matmul(X, self.W3.transpose(0,1))
        X_prob = torch.add(X_prob, torch.cat((self.h_eeg_bias,self.h_eye_bias),dim=-1))
        X_prob = torch.sigmoid(X_prob)
        sample_X_prob = self.sampling(X_prob)
        return X_prob, sample_X_prob

    def sampling(self, prob):
        s = torch.distributions.Bernoulli(prob).sample()
        return s

    def reconstruct_error_eeg(self, data):
        '''
        compute the reconstruction error for the data
        :param data:
        :return:
        '''
        return self.contrastive_divergence(data, False)

    def reconstruct_eeg(self,X,n_gibbs):
        '''
        This will reconstruct the EEG sample with k steps of gibbs sampling
        :param X:
        :param n_gibbs:
        :return:
        '''
        v = X
        for i in range(n_gibbs):
            prob_h, h = self.v2h_eeg(v)
            prob_v, v = self.h2v_eeg(prob_h)
        return prob_v, v

    def reconstruct_eye(self,X,n_gibbs):
        v = X
        for i in range(n_gibbs):
            prob_h, h = self.v2h_eye(v)
            prob_v, v = self.h2v_eye(prob_h)
        return prob_v, v

    def reconstruct_shared(self,X,n_gibbs):
        v = X
        for i in range(n_gibbs):
            prob_h, h = self.v2h_shared(v)
            prob_v, v = self.h2v_eeg(prob_h)
        return prob_v, v

    def constrastive_divergence_eeg(self, input_data, training=True, n_gibbs_sampling_steps=1, lr=0.001):
        # positive phase
        positive_hidden_probabilities, positive_hidden_act = self.v2h_eeg(input_data)

        positive_associations = torch.matmul(input_data.t(), positive_hidden_act)

        # negative phase
        hidden_activations = positive_hidden_act

        for i in range(int(n_gibbs_sampling_steps)):
            visible_probabilities, _ = self.h2v_eeg(hidden_activations)
            hidden_probabilities, hidden_activations = self.v2h_eeg(visible_probabilities)

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        if training:
            batch_size = self.batch_size

            g = (positive_associations - negative_associations)
            grad_update = g / batch_size
            v_bias_update = torch.sum(input_data - negative_visible_probabilities,dim=0) / batch_size
            h_bias_update = torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0) / batch_size

            self.W1.data = self.W1.data + lr * grad_update
            self.v_eeg_bias.data = self.v_eeg_bias.data + lr * v_bias_update
            self.h_eeg_bias.data = self.h_eeg_bias.data + lr * h_bias_update

        # Compute reconstruction error
        error = torch.mean(torch.sum((input_data - negative_visible_probabilities) ** 2, dim=0))
        return error, torch.sum(torch.abs(grad_update))

    def constrastive_divergence_eye(self, input_data, training=True, n_gibbs_sampling_steps=1, lr=0.001):
        # positive phase
        positive_hidden_probabilities, positive_hidden_act = self.v2h_eye(input_data)

        positive_associations = torch.matmul(input_data.t(), positive_hidden_act)

        # negative phase
        hidden_activations = positive_hidden_act

        for i in range(int(n_gibbs_sampling_steps)):
            visible_probabilities, _ = self.h2v_eye(hidden_activations)
            hidden_probabilities, hidden_activations = self.v2h_eye(visible_probabilities)

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        if training:
            batch_size = self.batch_size

            g = (positive_associations - negative_associations)
            grad_update = g / batch_size
            v_bias_update = torch.sum(input_data - negative_visible_probabilities, dim=0) / batch_size
            h_bias_update = torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0) / batch_size

            self.W2.data = self.W2.data + lr * grad_update
            self.v_eye_bias.data = self.v_eye_bias.data + lr * v_bias_update
            self.h_eye_bias.data = self.h_eye_bias.data + lr * h_bias_update

        # Compute reconstruction error
        error = torch.mean(torch.sum((input_data - negative_visible_probabilities) ** 2, dim=0))
        return error, torch.sum(torch.abs(grad_update))

    def constrastive_divergence_shared(self, input_data, training=True, n_gibbs_sampling_steps=1, lr=0.001):
        # positive phase
        positive_hidden_probabilities, positive_hidden_act = self.v2h_shared(input_data)

        positive_associations = torch.matmul(input_data.t(), positive_hidden_act)

        # negative phase
        hidden_activations = positive_hidden_act

        for i in range(int(n_gibbs_sampling_steps)):
            visible_probabilities, _ = self.h2v_shared(hidden_activations)
            hidden_probabilities, hidden_activations = self.v2h_shared(visible_probabilities)

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        if training:
            batch_size = self.batch_size

            g = (positive_associations - negative_associations)
            grad_update = g / batch_size
            v_bias_update = torch.sum(input_data - negative_visible_probabilities, dim=0) / batch_size
            h_bias_update = torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0) / batch_size

            self.W3.data = self.W3.data + lr * grad_update
            self.h_eeg_bias.data = self.h_eeg_bias.data + lr * v_bias_update[:self.hidden_units]
            self.h_eye_bias.data = self.h_eye_bias.data + lr * v_bias_update[self.hidden_units:]
            self.shared_feature_bias.data = self.shared_feature_bias.data + lr * h_bias_update

        # Compute reconstruction error
        error = torch.mean(torch.sum((input_data - negative_visible_probabilities) ** 2, dim=0))
        return error, torch.sum(torch.abs(grad_update))
    def encoder(self, X):
        eeg_hidden_probabilities, eeg_hidden_sampling = self.v2h_eeg(X[:, :310])
        eye_hidden_probabilities, eye_hidden_sampling = self.v2h_eye(X[:, 310:])
        shared_hidden_probabilities, shared_hidden_sampling = self.v2h_shared(
            torch.cat((eeg_hidden_probabilities, eye_hidden_probabilities), dim=1))
        return shared_hidden_probabilities, shared_hidden_sampling

    def decoder(self, X):
        shared_visible_probabilities = torch.sigmoid(torch.add(torch.matmul(X, self.W3_T),
                                                               torch.cat((self.h_eeg_bias_T, self.h_eye_bias_T),
                                                                         dim=-1)))
        eeg_hidden = shared_visible_probabilities[:, :self.hidden_units]
        eye_hidden = shared_visible_probabilities[:, self.hidden_units:]
        eeg_visible = torch.sigmoid(torch.add(torch.matmul(eeg_hidden, self.W1_T), self.v_eeg_bias_T))
        eye_visible = torch.sigmoid(torch.add(torch.matmul(eye_hidden, self.W2_T), self.v_eye_bias_T))
        return eeg_visible, eye_visible

    def forward(self, input_data):
        "data->shared"
        shared_representation, _ = self.encoder(input_data)
        reconstruct_eeg, reconstruct_eye = self.decoder(shared_representation)
        return reconstruct_eeg, reconstruct_eye

    def step(self,input_data, epoch, num_epochs,flag):
        """
        A train step to update related parameters
        :param input_Data: input feature
        :param epoch: current epoch
        :param num_epochs: total epoch
        :param flag: in  [eeg, eye, shared] means update which RBM
        :return:
        """
        if self.increase_to_cd_k:
            n_gibbs_sampling_steps = int(math.ceil((epoch / num_epochs) * self.k))
        else:
            n_gibbs_sampling_steps = self.k

        if self.learning_rate_decay:
            lr = self.learning_rate / epoch
        else:
            lr = self.learning_rate

        if flag == 'eeg':
            return self.constrastive_divergence_eeg(input_data,True,n_gibbs_sampling_steps,lr)
        elif flag == 'eye':
            return self.constrastive_divergence_eye(input_data,True,n_gibbs_sampling_steps,lr)
        elif flag == 'shared':
            return self.constrastive_divergence_shared(input_data,True,n_gibbs_sampling_steps,lr)

    def pre_train_phase(self, train_dataloader, num_epochs=50,batch_size=16):
        """
        the pretrain phase to fix the first three RBM: eeg, eye, shared
        :param train_dataloader: a iteration object implement dataloader interface
        :param num_epochs: total epochs of training
        :param batch_size: batch size
        :return:
        """
        self.batch_size = batch_size
        if (isinstance(train_dataloader, torch.utils.data.DataLoader)):
            train_loader = train_dataloader
        else:
            train_loader = torch.utils.data.DataLoader(train_dataloader, batch_size=batch_size)
        print("*" * 100)
        print("starting fitting two basic unimodal RBM ")
        for epoch in range(1,num_epochs+1):
            epoch_err = 0.
            n_batches = int(len(train_loader))

            cost_eeg = torch.FloatTensor(n_batches, 1)
            grad_eeg = torch.FloatTensor(n_batches, 1)
            cost_eye = torch.FloatTensor(n_batches, 1)
            grad_eye = torch.FloatTensor(n_batches, 1)

            for i, (batch, _) in tqdm(enumerate(train_loader), ascii=True, desc="EEG AND EYE RBM fitting", file=sys.stdout):
                #print(batch.shape)
                batch = batch.bernoulli()
                batch = Variable(batch)
                if self.use_gpu:
                    batch = batch.cuda()
                cost_eeg[i-1], grad_eeg[i-1] = self.step(batch[:,:310], epoch, num_epochs,flag='eeg')
                cost_eye[i - 1], grad_eye[i - 1] = self.step(batch[:,310:], epoch, num_epochs, flag='eye')
            print("*" * 100)
            print("Fitting info for EEG RBM in Epoch:{}, avg_cost = {}, std_cost = {}, avg_grad = {}, std_grad = {}".format(
                epoch, torch.mean(cost_eeg),torch.std(cost_eeg),torch.mean(grad_eeg),torch.std(grad_eeg)
            ))
            print("Fitting info for EYE RBM in Epoch:{}, avg_cost = {}, std_cost = {}, avg_grad = {}, std_grad = {}".format(
                    epoch, torch.mean(cost_eye), torch.std(cost_eye), torch.mean(grad_eye), torch.std(grad_eye)
                ))
            print("*" * 100)
        print("Finishing fitting two basic unimodal RBM, the parameters of those two model has been fixed ")
        print("*" * 100)

        print("\n\n\n")
        print("*" * 100)
        print("Starting fitting shared RBM")
        for epoch in range(num_epochs+1):
            n_batches = int(len(train_loader))

            cost_ = torch.FloatTensor(n_batches, 1)
            grad_ = torch.FloatTensor(n_batches, 1)

            for i, (batch, _) in tqdm(enumerate(train_loader),ascii=True,desc="Shared RBM layer Fitting",file=sys.stdout):
                batch = batch.bernoulli()
                batch = Variable(batch)
                if self.use_gpu:
                    batch = batch.cuda()
                _, eeg_hidden_act = self.v2h_eeg(batch[:, :310])
                _, eye_hidden_act = self.v2h_eye(batch[:, 310:])
                visible_shared = torch.cat((eeg_hidden_act,eye_hidden_act),dim=1)
                cost_[i-1], grad_[i-1] = self.step(visible_shared,epoch,num_epochs,flag='shared')
            print(
                "Fitting info for Shared RBM in Epoch:{}, avg_cost = {}, std_cost = {}, avg_grad = {}, std_grad = {}".format(
                    epoch, torch.mean(cost_), torch.std(cost_), torch.mean(grad_), torch.std(grad_)
                ))
        print("Finishing fitting Shared RBM")
        print("*" * 100)

        print("*" * 100)
        print("unfolding to BDAE...")
        # after training two layer RBM, uufolding the stacked RBMs into a bimodal deep autoencoder
        self.W1_T.copy_(self.W1.t())
        self.W2_T.copy_(self.W2.t())
        self.W3_T.copy_(self.W3.t())
        self.v_eeg_bias_T.copy_(self.v_eeg_bias)
        self.v_eye_bias_T.copy_(self.v_eye_bias)
        self.h_eeg_bias_T.copy_(self.h_eeg_bias)
        self.h_eye_bias_T.copy_(self.h_eye_bias)
        assert (self.W1_T.t() == self.W1).all() == 1, "parameter fix failure"
        print("Finishing unfolding to BDAE...")
        print("*" * 100)

        # Register gradient hook
        self.W1 = torch.nn.Parameter(self.W1)
        self.W2 = torch.nn.Parameter(self.W2)
        self.W3 = torch.nn.Parameter(self.W3)
        self.W1_T = torch.nn.Parameter(self.W1_T)
        self.W2_T = torch.nn.Parameter(self.W2_T)
        self.W3_T = torch.nn.Parameter(self.W3_T)
        self.v_eye_bias = torch.nn.Parameter(self.v_eye_bias)
        self.v_eeg_bias = torch.nn.Parameter(self.v_eeg_bias)
        self.h_eye_bias = torch.nn.Parameter(self.h_eye_bias)
        self.h_eeg_bias = torch.nn.Parameter(self.h_eeg_bias)
        self.v_eye_bias_T = torch.nn.Parameter(self.v_eye_bias_T)
        self.v_eeg_bias_T = torch.nn.Parameter(self.v_eeg_bias_T)
        self.h_eye_bias_T = torch.nn.Parameter(self.h_eye_bias_T)
        self.h_eeg_bias_T = torch.nn.Parameter(self.h_eeg_bias_T)
        self.shared_feature_bias = torch.nn.Parameter(self.shared_feature_bias)

    def fine_tune(self, train_dataloader, num_epochs, batch_size=16,savepath=None):
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
                epoch, running_loss, running_eeg_loss,running_eye_loss
            ))

            # 存储模型参数
            if running_loss < lowest_error:
                lowest_error = running_loss
                print("saving model...")

                torch.save(self, os.path.join(savepath, 'BDAE.pth'))


def evaluation(session, individual):
    restored_path = '../../saved_models/BDAE_egg_eye/%d/%d' % (session, individual)
    net = BDAE()

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

    # train_X = (train_X - train_X.min()) / (train_X.max() - train_X.min())
    # test_X = (test_X - test_X.min()) / (test_X.max() - test_X.min())
    # X = (X - X.min()) / (X.max() - X.min())

    max_ = train_X.max()
    min_ = train_X.min()
    train_X = (train_X - min_) / (max_ - min_)
    test_X = (test_X - min_) / (max_ - min_)
    X = (X - X.min()) / (X.max() - X.min())

    # construct dataloader
    # seed_iv_train_dataset = SEED_IV_DATASET(X=train_X, Y=train_Y)
    # seed_iv_test_dataset = SEED_IV_DATASET(X=test_X, Y=test_Y)
    seed_iv_dataset = SEED_IV_DATASET(X=X, Y=Y)
    train_dataloader = DataLoader(dataset=seed_iv_dataset,shuffle=True,batch_size=batch_size,num_workers=4)
    # seed_iv_train_loader = DataLoader(dataset=seed_iv_train_dataset, shuffle=False, batch_size=32, num_workers=4)
    # seed_iv_test_loader = DataLoader(dataset=seed_iv_test_dataset, shuffle=True, batch_size=32, num_workers=4)

    savepath = '../../saved_models/BDAE_egg_eye/%d/%d' % (session, individual)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # modal construction
    net = BDAE(visible_units_eeg=310, visible_units_eye=31, hidden_units=100, k=2, learning_rate=1e-3,
               xavier_init=True)
    if os.path.exists(os.path.join(savepath, 'BDAE.pth')):
        print("checkpoint founded, loading...")
        # loading model
        net = torch.load(os.path.join(savepath, 'BDAE.pth'))
        print("loading checkpoint successful")
    else:
        print("checkpoint hasn't been founded, rebuilding model...")
        net = BDAE(visible_units_eeg=310, visible_units_eye=31, hidden_units=50, k=2, learning_rate=1e-3,
                   xavier_init=True)
        net.pre_train_phase(train_dataloader=train_dataloader,num_epochs=50,batch_size=batch_size)
        net.fine_tune(train_dataloader=train_dataloader,num_epochs=50,batch_size=batch_size,savepath=savepath)
        print("BDAE model rebuilt and training complete...")
    # compare the discriminative capability of two kind feature
    from Unimodal_Recognition.svm_classifier import svm_classifier
    from  sklearn.svm import SVC
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
    # simple fusion concat feature
    sf_svm = SVC(kernel='rbf',C=1)
    # BDAE fusion feature
    bdae_svm = SVC(kernel='rbf',C=1)
    sf_svm.fit(train_X, train_Y)
    print("1")
    sf_acc = accuracy_score(test_Y, sf_svm.predict(test_X))
    print(classification_report(test_Y, sf_svm.predict(test_X)))
    print(confusion_matrix(test_Y, sf_svm.predict(test_X)))
    print("2")
    train_X_bdae = torch.Tensor(train_X)
    test_X_bdae = torch.Tensor(test_X)
    with torch.no_grad():
        Train = net.encoder(train_X_bdae)[0].numpy()
        Test = net.encoder(test_X_bdae)[0].numpy()
    print(train_X[0])
    print(Train[0])
    print("5")
    bdae_svm.fit(Train, train_Y)
    print("3")
    bdae_acc = accuracy_score(test_Y,bdae_svm.predict(Test))
    print(classification_report(test_Y,bdae_svm.predict(Test)))
    print(confusion_matrix(test_Y,bdae_svm.predict(Test)))
    print("4")
    print("simple feature fusion:{}\nBDAE feature fusion:{}\n\tdiffer{}".format(
        sf_acc, bdae_acc, sf_acc-bdae_acc
    ))


if __name__=='__main__':
    main()
    # net = BDAE(visible_units_eeg=310,visible_units_eye=31,hidden_units=50,k=2,learning_rate=1e-3,xavier_init=True)
    # print(net.parameters())
    # for params in net.parameters():
    #     print(params)

