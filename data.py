import torch
from torch import distributions
import gin

@gin.configurable
class Data:
    def __init__(self, dataset_name, num_samples, alpha=1., noise_dim=0, seed=0):
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.alpha = alpha
        self.noise_dim = noise_dim
        self.seed = seed

        torch.manual_seed(self.seed)

        if self.dataset_name == "linear":
            self.get_dset_linear()
        elif self.dataset_name == "helix":
            self.get_dset_helix()
        elif self.dataset_name == "sphere":
            self.get_dset_sphere()
        else:
            raise NotImplementedError("No dataset named \"{}\" is implemented".format(self.dataset_name))

        self.add_noise()

    def get_dset_linear(self):
        self.distr_z = distributions.Uniform(-self.alpha/2, self.alpha/2)
        self.var_z = self.distr_z.sample(sample_shape=[1, self.num_samples])
        self.distr_x = distributions.Uniform(0, 1)
        self.var_x = self.distr_x.sample(sample_shape=[1, self.num_samples])
        self.var_y = self.var_x + self.var_z

    def get_dset_helix(self):
        raise NotImplementedError

    def get_dset_sphere(self):
        raise NotImplementedError

    def add_noise(self):
        #TODO: add noise to var_x, var_y (and an option to select noise distribution?)
        self.distr_noisedim = distributions.Normal(0, 1)

        self.var_x = torch.cat(
            [self.var_x, self.distr_noisedim.rsample(sample_shape=[self.noise_dim, self.num_samples])], axis=0)
        self.var_y = torch.cat(
            [self.var_y, self.distr_noisedim.rsample(sample_shape=[self.noise_dim, self.num_samples])], axis=0)