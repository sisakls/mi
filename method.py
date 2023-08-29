import wandb

class Method:
    def __init__(self, method_name, seed):
        self.method_name = method_name
        self.seed = seed

        if self.method_name == "test":
            self.eval = test

        else: 
            raise Exception("No method named \"{}\" is implemented".format(self.method_name))

def test(var_X, var_Y):
    if (var_X >= 0 and var_Y >= 0) or (var_X <= 0 and var_Y <= 0):
        MI = 1
    else:
        MI = 0
    wandb.log({"MI": MI})