import random

import numpy as np

from .data import *
from .metrics import *


def get_device(device):
    """返回指定的GPU设备

    :param device: int GPU编号，-1表示CPU
    :return: torch.device
    """
    return torch.device(f'cuda:{device}' if device >= 0 and torch.cuda.is_available() else 'cpu')
