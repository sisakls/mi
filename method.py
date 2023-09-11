import gin
import wandb

#import numpy as np
import torch
#import math
#from scipy.spatial import distance
from scipy.special import loggamma

@gin.configurable
class Method:
    def __init__(self, method_name, seed):
        self.method_name = method_name
        self.seed = seed

        if self.method_name == "KL": 
            self.eval = self.KL_estimator #Kozachenko-Leonenko estimator
        else: 
            raise NotImplementedError("No method named \"{}\" is implemented".format(self.method_name))

    def KL_estimator(self, var_x, var_y, p_norm=2, eps=1e-10):
        var_x = self.normalize_input(var_x)
        var_y = self.normalize_input(var_y)
        var_joint = torch.cat([var_x, var_y], axis=0)
        MI = (
            self.kNN_entropy(var_x, 1, p_norm=p_norm, eps=eps) 
            + self.kNN_entropy(var_y, 1, p_norm=p_norm, eps=eps) 
            - self.kNN_entropy(var_joint, 1, p_norm=p_norm, eps=eps))
        wandb.log({"MI": MI})

    def kNN_entropy(self, var_, k, p_norm=2, eps=1e-10):
        dim = var_.shape[0]
        radius = self.kNN_radius(var_, k, p_norm=p_norm, eps=eps)
        log_mean_volume = (
            (torch.log(radius)).mean() 
            + (dim/2) * torch.log(torch.tensor(3.1416)) 
            - loggamma(dim/2 + 1) 
            + 0.5772) #Euler-Mascheroni constant
        return log_mean_volume 

    def kNN_radius(self, var_, k, p_norm=2, eps=1e-10):
        dist = torch.cdist(var_.T, var_.T, p=p_norm)
        dist = dist.sort().values
        return dist[:, k] + eps

    def normalize_input(self, var_): #Z-score normalization
        var_ = var_ - var_.mean(axis=1, keepdim=True)
        var_ = var_ / torch.std(var_, axis=1, keepdim=True)
        return var_