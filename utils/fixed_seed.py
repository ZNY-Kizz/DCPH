import random
import numpy as np
import torch

def SetSeed():
    random.seed(613)
    np.random.seed(613)
    torch.manual_seed(613)
    torch.cuda.manual_seed(613)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)