import numpy as np
import gin
import wandb
from utils import gin_config_to_dict

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

    def estimate(self):
        self.method.eval(
            exp_manager.data.var_X,
            exp_manager.data.var_Y)

wandb.init(project="mi", entity="sisaklsanyo")
wandb.config.update(gin_config_to_dict(gin.config_str()))

exp_manager = ExperimentManager()
exp_manager.estimate()

wandb.finish()

print(exp_manager.__dict__)