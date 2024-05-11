import torch
from absl import app, flags
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
        for sample_size in [100, 300, 1000, 3000, 10000]:
            self.method.eval(
                self.data.var_x[:sample_size, :],
                self.data.var_y[:sample_size, :])

def main(argv):
    #parse gin file
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)

    #set up wandb
    wandb.init(project="mi", entity="sisaklsanyo")
    run_counter = wandb.run.name.split("-")[-1]
    wandb.run.name = run_counter + "-" + FLAGS.gin_file[0].split('/')[-1][:-4]
    wandb.config.update(gin_config_to_dict(gin.config_str()))

    #run experiment
    exp_manager = ExperimentManager()
    exp_manager.estimate()

    wandb.finish()

#set up gin
if __name__ == '__main__':
    flags.DEFINE_multi_string('gin_file', None, "List of paths to the config files.")
    flags.DEFINE_multi_string('gin_param', None, "Newline separated list of Gin param bindings.")
    FLAGS = flags.FLAGS

    app.run(main)