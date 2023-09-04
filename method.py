import gin
import wandb

import numpy as np
from scipy.spatial import distance
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

    def KL_estimator(self, var_x, var_y, metric="euclidean", eps=1e-10):
        var_x = self.normalize_input(var_x)
        var_y = self.normalize_input(var_y)
        var_joint = np.concatenate([var_x, var_y], axis=1)
        MI = (
            self.kNN_entropy(var_x, 1, metric=metric, eps=eps) + 
            self.kNN_entropy(var_y, 1, metric=metric, eps=eps) - 
            self.kNN_entropy(var_joint, 1, metric=metric, eps=eps))
        wandb.log({"MI": MI})

    def kNN_entropy(self, var_, k, metric="euclidean", eps=1e-10):
        dim = var_.shape[-1]
        radius = self.kNN_radius(var_, k, metric=metric, eps=eps)
        log_mean_volume = np.mean(np.log(radius)) + (dim/2)*np.log(np.pi) - loggamma(dim/2 + 1)
        return log_mean_volume + 0.57721

    def kNN_radius(self, var_, k, metric="euclidian", eps=1e-10):
        dist = distance.cdist(var_, var_, metric=metric)
        dist = np.sort(dist, -1)
        return dist[:, k] + eps

    def normalize_input(self, var_):
        var_ = var_ - np.mean(var_, axis=0)
        var_ = var_ - np.linalg.norm(var_, axis=0)
        return var_