import os
import math
import numpy as np
import torch

from einops import repeat
from torch import nn


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * len(x_shape) - 1))