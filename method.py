import gin
import wandb

import torch
#from scipy.spatial import distance
from scipy.special import loggamma, psi
from sympy import harmonic
from mi_estimators import *

@gin.configurable
class Method:
    def __init__(
        self, method_name, seed, k=1, eps=1e-10, MI=True, 
        hidden_size=15, learning_rate=0.005, num_iters=100):

        self.method_name = method_name
        self.seed = seed
        #if type(k) == list: self.k_list = k
        #elif type(k) == int: self.k_list = [k]
        #else: raise TypeError
        #if type(eps) == list: self.eps_list = eps
        #elif type(eps) == float: self.eps_list = [eps]
        #else: raise TypeError
        self.k = k
        self.eps = eps
        self.MI = MI
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.step = 0

        if self.method_name == "kNN": 
            self.eval = self.kNN_estimator
        elif self.method_name == "KSG1":
            self.eval = self.KSG1_estimator
        elif self.method_name == "KSG2":
            self.eval = self.KSG2_estimator
        elif self.method_name in ["NWJ", "MINE", "InfoNCE","L1OutUB","CLUB","CLUBSample"]:
            self.eval = self.neural_eval
        else: 
            raise NotImplementedError("No method named \"{}\" is implemented".format(self.method_name))


    #def kNN_eval(self, var_x, var_y):
    #    for eps in self.eps_list:
    #        for k in self.k_list:
    #            self.kNN_estimator(var_x, var_y, k=k, eps=eps)


    def neural_eval(self, var_x, var_y):
        dim, N = var_x.shape
        model = eval(self.method_name)(dim, dim, self.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        mi_est_values = []

        for global_iter in range(self.num_iters):
            #batch_x, batch_y = correlated_linear(alpha=0.01, dim=sample_dim, batch_size=batch_size)
            model.eval()
            mi_est_values.append(model(var_x.T, var_y.T).item())
            wandb.log({"MI": mi_est_values[-1]}, step=self.step)
            self.step += 1
            
            model.train() 
            model_loss = model.learning_loss(var_x.T, var_y.T)
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()
            #del batch_x, batch_y
            #torch.cuda.empty_cache()


    def kNN_estimator(self, var_x, var_y, p_norm=2):
        #var_x = self.normalize_input(var_x)
        #var_y = self.normalize_input(var_y)
        var_joint = torch.cat([var_x, var_y], axis=0)
        H_x = self.kNN_entropy(var_x, self.k, p_norm=p_norm, eps=self.eps) 
        H_y = self.kNN_entropy(var_y, self.k, p_norm=p_norm, eps=self.eps) 
        H_xy = self.kNN_entropy(var_joint, self.k, p_norm=p_norm, eps=self.eps)

        if self.MI: wandb.log({"MI": (H_x + H_y - H_xy).item()})#, step=self.step)
        else: wandb.log({"H_x": H_x.item(), "H_y": H_y.item(), "H_xy": H_xy.item()})#, step=self.step)
        #wandb.log({"epsilon": eps, "k": k}, step=self.step)
        self.step += 1


    def KSG1_estimator(self, var_x, var_y, p_norm=2):
        var_joint = torch.cat([var_x, var_y], axis=0)
        n_samples = var_x.shape[1]
        dist_x = self.kNN_radius(var_x, None, p_norm=p_norm)
        dist_y = self.kNN_radius(var_y, None, p_norm=p_norm)
        dist_joint = torch.maximum(dist_x, dist_y).sort().values[:, self.k] + self.eps

        n_x = self.KSG_count(var_x, dist_joint, p_norm=p_norm)
        n_y = self.KSG_count(var_y, dist_joint, p_norm=p_norm)

        MI = (
            (torch.log(n_x) + torch.log(n_y)).mean() 
            + torch.log(torch.tensor([n_samples])) 
            + torch.log(torch.tensor([self.k])))

        wandb.log({"MI": MI.item()})#, step=self.step)
        self.step += 1


    def KSG2_estimator(self, var_x, var_y, p_norm=2):
        var_joint = torch.cat([var_x, var_y], axis=0)
        #dist = TODO
        n_samples = var_x.shape[1]

        raise NotImplementedError("KSG2 not yet implemented")


    def kNN_entropy(self, var_, k=1, p_norm=2, eps=1e-10):
        dim, n_samples = var_.shape
        radius = self.kNN_radius(var_, k, p_norm=p_norm, eps=eps)
        log_mean_volume = (
            torch.log(torch.tensor([n_samples]))
            + (dim * torch.log(radius)).mean() 
            + (dim/2) * torch.log(torch.tensor(3.1416)) 
            - psi(k)
            - loggamma(dim/2 + 1))
        return log_mean_volume


    def kNN_radius(self, var_, k=1, p_norm=2, eps=1e-10):
        dist = torch.cdist(var_.T, var_.T, p=p_norm)
        dist = dist.sort().values
        if k is not None:
            return dist[:, k] + eps
        else:
            return dist

    
    def KSG_count(self, var_, radii, p_norm=2):
        n_samples = var_.shape[1]
        dist_ = self.kNN_radius(var_, None, p_norm=p_norm)
        count = torch.zeros(n_samples) #counting points that fall in the given radius
        for i in range(n_samples): 
            j = self.k #at least k points fall in the radius
            while dist_[i,j] < radii[i]: j+=1
            count[i] = j
        return count


    def normalize_input(self, var_): #Z-score normalization
        var_ -= var_.mean(axis=1, keepdim=True)
        var_ /= torch.std(var_, axis=1, keepdim=True)
        return var_