import gin
import wandb

import torch
from torch import distributions
from numpy import log

@gin.configurable
class Data:
    def __init__(
        self, dataset_name, num_samples, dim=1, seed=0, 
        noise_dim=0, alpha=1., precov_mtx=[[1., 0.], [0.5, 1.]]):
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.dim = dim
        self.seed = seed
        self.noise_dim = noise_dim
        self.alpha = alpha
        self.precov_mtx = precov_mtx #must be lower triangular, diagonal entries all positive (or None)
        
        torch.manual_seed(self.seed)

        if self.dataset_name == "linear":
            self.get_dset_linear()
        elif self.dataset_name == "gaussian":
            self.get_dset_gaussian()
        elif self.dataset_name == "helix":
            self.get_dset_helix()
        elif self.dataset_name == "sphere":
            self.get_dset_sphere()
        else:
            raise NotImplementedError("No dataset named \"{}\" is implemented".format(self.dataset_name))

        self.add_noise()


    def get_dset_linear(self):
        self.distr_z = distributions.Uniform(-self.alpha/2, self.alpha/2)
        self.var_z = self.distr_z.sample(sample_shape=[self.num_samples, self.dim])
        self.distr_x = distributions.Uniform(0, 1)
        self.var_x = self.distr_x.sample(sample_shape=[self.num_samples, self.dim])
        self.var_y = self.var_x + self.var_z

        wandb.log({"True MI": self.dim * (self.alpha/2 - log(self.alpha))})


    def get_dset_gaussian(self):
        if self.precov_mtx == None:
            self.precov_mtx = torch.tril(2*torch.rand([2*self.dim, 2*self.dim]) - 1)
            for i in range(2*self.dim):
                self.precov_mtx[i,i] = 1
        else:
            self.precov_mtx = torch.tensor(self.precov_mtx)
        assert self.precov_mtx.shape == torch.Size([2*self.dim, 2*self.dim])

        cov_mtx = self.precov_mtx @ self.precov_mtx.T
        self.distr_xy = distributions.MultivariateNormal(torch.zeros(2*self.dim), cov_mtx)
        self.var_xy = self.distr_xy.sample(sample_shape=[self.num_samples])
        self.var_x, self.var_y = self.var_xy[:, :self.dim], self.var_xy[:, self.dim:]

        mtx_x, mtx_y = cov_mtx[:self.dim, :self.dim], cov_mtx[self.dim:, self.dim:]
        wandb.log({"True MI": 0.5 * (torch.log(torch.det(mtx_x) * torch.det(mtx_y)) - torch.log(torch.det(cov_mtx))).item()})


    def get_dset_helix(self):
        raise NotImplementedError

    def get_dset_sphere(self):
        raise NotImplementedError

    def add_noise(self):
        #TODO: add noise to var_x, var_y (and an option to select noise distribution?)
        self.distr_noisedim = distributions.Normal(0, 1)

        self.var_x = torch.cat(
            [self.var_x, self.distr_noisedim.rsample(sample_shape=[self.num_samples, self.noise_dim])], axis=1)
        self.var_y = torch.cat(
            [self.var_y, self.distr_noisedim.rsample(sample_shape=[self.num_samples, self.noise_dim])], axis=1)