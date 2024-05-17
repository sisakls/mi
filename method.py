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
        self, method_name, seed, MI=True, 
        k=1, eps=1e-10, p_norm=2,
        hidden_size=15, learning_rate=0.005, num_iters=100):

        self.method_name = method_name
        self.seed = seed
        self.MI = MI

        self.k = k
        self.eps = eps
        self.p_norm = float(p_norm)

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.step = 0
        self.trainstep = 0

        nearest_neighbor_estimators = {
            "kNN": self.kNN_estimator, "KSG1": self.KSG1_estimator, "KSG2": self.KSG2_estimator}
        if self.method_name in nearest_neighbor_estimators.keys(): 
            self.eval = nearest_neighbor_estimators[self.method_name]
        elif self.method_name in ["NWJ", "MINE", "InfoNCE","L1OutUB","CLUB","CLUBSample"]:
            self.eval = self.neural_eval
        else: 
            raise NotImplementedError("No method named \"{}\" is implemented".format(self.method_name))


    def neural_eval(self, var_x, var_y):
        dim = var_x.shape[1]
        model = eval(self.method_name)(dim, dim, self.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        mi_est_values = []

        for global_iter in range(self.num_iters):
            #batch_x, batch_y = correlated_linear(alpha=0.01, dim=sample_dim, batch_size=batch_size)
            model.eval()
            mi_est_values.append(model(var_x, var_y).item())
            wandb.log({"train MI": mi_est_values[-1]}, step=self.trainstep)
            self.trainstep += 1
            
            model.train() 
            model_loss = model.learning_loss(var_x, var_y)
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()
            #del batch_x, batch_y
            #torch.cuda.empty_cache()
        wandb.log({"MI": mi_est_values[-1]})#, step=self.step)

    def kNN_estimator(self, var_x, var_y):
        var_joint = torch.cat([var_x, var_y], axis=1)
        H_x = self.kNN_entropy(var_x, self.k, p_norm=self.p_norm, eps=self.eps) 
        H_y = self.kNN_entropy(var_y, self.k, p_norm=self.p_norm, eps=self.eps) 
        H_xy = self.kNN_entropy(var_joint, self.k, p_norm=self.p_norm, eps=self.eps)

        if self.MI: wandb.log({"MI": (H_x + H_y - H_xy).item()})#, step=self.step)
        else: wandb.log({"H_x": H_x.item(), "H_y": H_y.item(), "H_xy": H_xy.item()})#, step=self.step)
        #wandb.log({"epsilon": eps, "k": k}, step=self.step)
        self.step += 1


    def KSG1_estimator(self, var_x, var_y):
        n_samples = var_x.shape[0]
        var_joint = torch.cat([var_x, var_y], axis=1)
        dist_x = self.kNN_radius(var_x, None, p_norm=self.p_norm, sorting=False)
        dist_y = self.kNN_radius(var_y, None, p_norm=self.p_norm, sorting=False)
        dist_joint = torch.maximum(dist_x, dist_y).sort().values[:, self.k] + self.eps

        n_x = self.KSG_count(var_x, dist_joint, p_norm=self.p_norm)
        n_y = self.KSG_count(var_y, dist_joint, p_norm=self.p_norm)

        MI = (
            - (torch.log(n_x) + torch.log(n_y)).mean() 
            + torch.log(torch.tensor([n_samples])) 
            + torch.log(torch.tensor([self.k])))

        wandb.log({"MI": MI.item()})#, step=self.step)
        self.step += 1


    def KSG2_estimator(self, var_x, var_y):
        n_samples = var_x.shape[0]
        var_joint = torch.cat([var_x, var_y], axis=1)
        dist_x = self.KSG_radius(var_x, k=None, p_norm=self.p_norm)
        dist_y = self.KSG_radius(var_y, k=None, p_norm=self.p_norm)
        dist_joint = torch.maximum(dist_x, dist_y)
        knn = dist_joint.topk(k=self.k+1, dim=1, largest=False)

        radii_x = var_x[knn.indices] - var_x[:,None,:].expand(var_x[knn.indices].shape)
        radii_x = radii_x.norm(p=self.p_norm, dim=2).max(dim=1).values + self.eps
        radii_y = var_y[knn.indices] - var_y[:,None,:].expand(var_y[knn.indices].shape)
        radii_y = radii_y.norm(p=self.p_norm, dim=2).max(dim=1).values + self.eps

        n_x = self.KSG_count(var_x, radii_x, p_norm=self.p_norm)
        n_y = self.KSG_count(var_y, radii_y, p_norm=self.p_norm)

        MI = (
            - (torch.log(n_x) + torch.log(n_y)).mean() 
            + torch.log(torch.tensor([n_samples])) 
            + torch.log(torch.tensor([self.k])))

        wandb.log({"MI": MI.item()})#, step=self.step)
        self.step += 1


    def kNN_entropy(self, var_, k=1, p_norm=2, eps=1e-10):
        n_samples, dim = var_.shape
        radius = self.kNN_radius(var_, k, p_norm=p_norm, eps=eps)
        if p_norm < float('inf'): #Unit ball hypervolume, L_p norm
            unit_V = (
                dim * (loggamma(1 + 1/p_norm) + torch.log(torch.tensor([2])))
                - loggamma(dim/p_norm + 1))
        elif p_norm == float('inf'): #Unit ball hypervolume, L_infty norm
            unit_V = dim * torch.log(torch.tensor([2]))
        log_mean_volume = (
            torch.log(torch.tensor([n_samples])) #log(n)/psi(n)
            - psi(k) #-log(k)/-psi(k)
            + (dim * torch.log(radius)).mean() #(dim/n)sum r_i
            + unit_V)
        return log_mean_volume


    def kNN_radius(self, var_, k=1, p_norm=2, eps=1e-10, sorting=True):
        dist = torch.cdist(var_, var_, p=p_norm)
        if sorting:
            dist = dist.sort().values
        if k is not None:
            return dist[:, k] + eps
        else:
            return dist


    def KSG_radius(self, var_, k=1, p_norm=2):
        n_samples = var_.shape[0]
        varvar = var_[:, :, None].repeat(1, 1, n_samples)
        dist = varvar - varvar.transpose(0,2)
        dist = dist.norm(p=p_norm, dim=1)
        if k is not None:
            return dist.topk(k+1, dim=0, largest=False)
        else:
            return dist

    
    def KSG_count(self, var_, radii, p_norm=2):
        n_samples = var_.shape[0]
        dist_ = self.kNN_radius(var_, None, p_norm=p_norm)
        count = torch.zeros(n_samples) #counting points that fall in the given radius
        for i in range(n_samples): 
            j = self.k #at least k points fall in the radius
            while dist_[i,j] < radii[i]: j+=1
            count[i] = j
        return count


    def normalize_input(self, var_): #Z-score normalization
        var_ -= var_.mean(axis=0, keepdim=True)
        var_ /= torch.std(var_, axis=0, keepdim=True)
        return var_