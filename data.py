import numpy as np

class Data:
    def __init__(self, dataset_name, seed):
        self.dataset_name = dataset_name
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)

        if self.dataset_name == "test":
            self.var_Z = self.rng.normal(0, 2, size=6)
            self.noise = self.rng.normal(0, 0.2, size=2)

            self.var_XX = sum(self.var_Z[:5])
            self.var_X = self.var_XX + self.noise[0]
            self.var_YY = sum(self.var_Z[-4:])
            self.var_Y = self.var_XX + self.noise[1]
        
        else:
            raise Exception("No dataset named \"{}\" is implemented".format(self.dataset_name))