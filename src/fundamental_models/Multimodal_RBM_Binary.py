from RBM import RBM
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import math
import numpy as np
import pickle
import codecs
class DATASET(Dataset):
    def __init__(self,x):
        self.data = x
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, item):
        return self.data[item]


with codecs.open('../../multimodal_feature_extracted/SEED_IV/1/fusion_feature.pk','rb') as f:
    DATA = pickle.load(f)

indis = DATA.keys()
data = DATA[list(indis)[1]]

TRAIN = data[:17]
TEST = data[17:]

for i, item in enumerate(TRAIN):
    if i == 0:
        train = item
    else:
        train = np.vstack((train, item))
for i, item in enumerate(TEST):
    if i == 0:
        test = item
    else:
        test = np.vstack((test, item))

eeg_train_data = train[:, :310]
eeg_test_data = test[:, :310]
eye_train_data = train[:, 310:-1]
eye_test_data = test[:, 310:-1]
train_Y = train[:, -1]
test_Y = test[:, -1]

eeg_train_data = (eeg_train_data - eeg_train_data.min()) / (eeg_train_data.max() - eeg_train_data.min())
eeg_test_data = (eeg_test_data - eeg_test_data.min()) / (eeg_test_data.max() - eeg_test_data.min())
eye_train_data = (eye_train_data - eye_train_data.min()) / (eye_train_data.max() - eye_train_data.min())
eye_test_data = (eye_test_data - eye_test_data.mean()) / (eye_test_data.max() - eye_test_data.min())

# eeg_loader = DataLoader(dataset=DATASET(torch.Tensor(eeg_train_data).bernoulli()), batch_size=32, shuffle=True, num_workers=0)
# eye_loader = DataLoader(dataset=DATASET(torch.Tensor(eye_train_data).bernoulli()), batch_size=32, shuffle=True, num_workers=0)

eeg_loader = DataLoader(dataset=torch.utils.data.TensorDataset(torch.Tensor(eeg_train_data).bernoulli(),torch.Tensor(train_Y)), batch_size=32, shuffle=True, num_workers=0)
eye_loader = DataLoader(dataset=torch.utils.data.TensorDataset(torch.Tensor(eye_train_data).bernoulli(),torch.Tensor(train_Y)), batch_size=32, shuffle=True, num_workers=0)

visible__eeg_units = 310
visible__eye_units = 31
hidden_units = 50
k = 2
learning_rate = 0.001
learning_rate_decay = True
xavier_init = True
increas_to_cd_k = False
use_gpu = False

rbm_eeg = RBM(visible__eeg_units, hidden_units, int(k), learning_rate, learning_rate_decay, xavier_init, increas_to_cd_k, use_gpu)
rbm_eye = RBM(visible__eye_units, hidden_units, int(k), learning_rate, learning_rate_decay, xavier_init, increas_to_cd_k, use_gpu)

epochs = 30
batch_size = 32
rbm_eeg.train(eeg_loader, epochs, batch_size)
rbm_eye.train(eye_loader, epochs, batch_size)
if not os.path.exists('../../saved_models/RBM'):
    os.makedirs('../../saved_models/RBM')
torch.save(rbm_eeg, '../../saved_models/RBM/rbm_eeg')
torch.save(rbm_eye, '../../saved_models/RBM/rbm_eye')


