"""
author: ouyangtianxiong
date:19/12/04
des:RBM With GaussianBernoulli
"""
from RBM import RBM
import torch
class GaussianBernoulliRBM(RBM):
    """
    Visible later can assume real value
    Hidden layer assumes Binary values only
    """
    def to_visible(self,x):
        """
        this visible units follow gaussian distribution
        :param x: the torch tensor with shape = (n_samples, features)
        :return: X_prob the new constructed layers (probabilities) sample_X_prob - sample of new layer (Gibbs Sampling)
        """
        X_prob = torch.matmul(x, self.W.transpose(0, 1))
        X_prob = torch.add(X_prob, self.v_bias)
        sample_X_prob = X_prob + torch.randn(X_prob.shape)

        return X_prob, sample_X_prob
