import torch
import gin
import wandb

import data
import method
from utils import gin_config_to_dict

@gin.configurable
class ExperimentManager:
    def __init__(self, seed=0):
        self.seed = seed
        self.data = data.Data(seed=self.seed)
        self.method = method.Method(seed=self.seed)

    def estimate(self):
        self.method.eval(
            exp_manager.data.var_x,
            exp_manager.data.var_y)

gin.parse_config_file("test.gin")

wandb.init(project="mi", entity="sisaklsanyo")
wandb.config.update(gin_config_to_dict(gin.config_str()))

exp_manager = ExperimentManager()
exp_manager.estimate()

wandb.finish()

#print(exp_manager.__dict__)