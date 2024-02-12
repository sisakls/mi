import gin
import wandb

import torch
#from scipy.spatial import distance
from scipy.special import loggamma
from sympy import harmonic

@gin.configurable
class Method:
    def __init__(self, method_name, seed, k=1, eps=1e-10, MI=True):
        self.method_name = method_name
        self.seed = seed
        if type(k) == list: self.k_list = k
        elif type(k) == int: self.k_list = [k]
        else: raise TypeError
        if type(eps) == list: self.eps_list = eps
        elif type(eps) == float: self.eps_list = [eps]
        else: raise TypeError
        self.MI = MI
        self.step = 0

        if self.method_name == "kNN": 
            self.eval = self.kNN_eval
        else: 
            raise NotImplementedError("No method named \"{}\" is implemented".format(self.method_name))

    def kNN_eval(self, var_x, var_y):
        for eps in self.eps_list:
            for k in self.k_list:
                self.kNN_estimator(var_x, var_y, k=k, eps=eps)
                self.step += 1

    def kNN_estimator(self, var_x, var_y, k=1, p_norm=2, eps=1e-10):
        #var_x = self.normalize_input(var_x)
        #var_y = self.normalize_input(var_y)
        var_joint = torch.cat([var_x, var_y], axis=0)
        H_x = self.kNN_entropy(var_x, k, p_norm=p_norm, eps=eps) 
        H_y = self.kNN_entropy(var_y, k, p_norm=p_norm, eps=eps) 
        H_xy = self.kNN_entropy(var_joint, k, p_norm=p_norm, eps=eps)
        if self.MI: wandb.log({"MI": H_x + H_y - H_xy}, step=self.step)
        else: wandb.log({"H_x": H_x, "H_y": H_y, "H_xy": H_xy}, step=self.step)
        wandb.log({"epsilon": eps, "k": k}, step=self.step)

    def kNN_entropy(self, var_, k=1, p_norm=2, eps=1e-10):
        dim, N = var_.shape
        radius = self.kNN_radius(var_, k, p_norm=p_norm, eps=eps)
        log_mean_volume = (
            torch.log(torch.tensor([N]))
            + (dim * torch.log(radius)).mean() 
            + (dim/2) * torch.log(torch.tensor(3.1416)) 
            - float(harmonic(k-1))
            - loggamma(dim/2 + 1) 
            + 0.5772) #Euler-Mascheroni constant
        return log_mean_volume

    def kNN_radius(self, var_, k=1, p_norm=2, eps=1e-10):
        dist = torch.cdist(var_.T, var_.T, p=p_norm)
        dist = dist.sort().values
        return dist[:, k] + eps

    def normalize_input(self, var_): #Z-score normalization
        var_ -= var_.mean(axis=1, keepdim=True)
        var_ /= torch.std(var_, axis=1, keepdim=True)
        return var_