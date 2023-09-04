import numpy as np
import gin

@gin.configurable
class Data:
    def __init__(self, dataset_name, num_samples, alpha=1., noise_dim=0, seed=0):
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.alpha = alpha
        self.noise_dim = noise_dim
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)

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
        self.var_z = self.rng.uniform(-self.alpha/2, self.alpha/2, size=[self.num_samples, 1])
        self.var_x = self.rng.uniform(0, 1, size=[self.num_samples, 1])
        self.var_y = self.var_x + self.var_z

    def get_dset_helix(self):
        raise NotImplementedError

    def get_dset_sphere(self):
        raise NotImplementedError

    def add_noise(self):
        #TODO: add noise to var_x, var_y (and an option to select noise distribution?)

        self.var_x = np.concatenate(
            [self.var_x, self.rng.normal(0, 1, size=[self.num_samples, self.noise_dim])], -1) #noise dimensions
        self.var_y = np.concatenate(
            [self.var_y, self.rng.normal(0, 1, size=[self.num_samples, self.noise_dim])], -1)