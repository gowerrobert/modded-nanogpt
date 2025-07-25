import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR
import warnings
from typing import Tuple
from .muon_polar import Muon
from .muon_nano import MuonNano
# from .sps import SPS
# from .adabound import AdaBoundW
# from .adabelief import AdaBelief
# from .lion import Lion

def get_optimizer(opt_config: dict) -> Tuple[torch.optim.Optimizer, dict]:
    """
    Main function mapping opt configs to an instance of torch.optim.Optimizer and a dict of hyperparameter arguments (lr, weight_decay,..).  
    For all hyperparameters which are not specified, we use PyTorch default.
    """
    
    name = opt_config['name']
    
    
    if name == 'sgd':
        opt_obj = torch.optim.SGD
        hyperp = {'lr': opt_config.get('lr', 0.001),
                  'weight_decay': opt_config.get('weight_decay', 0)
                  }
    elif 'muon-nano' ==name:
        opt_obj = MuonNano
        hyperp = {'lr': opt_config.get('lr', 0.05),
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  }
    elif 'muon' in name:
        opt_obj = Muon
        hyperp = {'lr': opt_config.get('lr', 0.05), 
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  }
    else:
        raise KeyError(f"Unknown optimizer name {name}.")
        
    return opt_obj, hyperp
