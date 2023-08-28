import numpy as np
import gin
import wandb

import data
import method

@gin.configurable
class ExperimentManager:
    def __init__(self, dataset_name, method_name, seed=0):
        self.seed = seed
        self.dataset_name = dataset_name
        self.method_name = method_name

        self.data = data.Data(
            self.dataset_name, 
            self.seed)
        self.method = method.Method(
            self.method_name, 
            self.seed)

exp_manager = ExperimentManager("test", "test")
exp_manager.method.eval(
    exp_manager.data.var_X,
    exp_manager.data.var_Y
)

print(exp_manager.__dict__)