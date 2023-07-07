import copy
import functools
import os
import numpy as np
import torch as th
import torch.distributed as dist
from torch.optim import AdamW

from . import dist_util, logger
from model.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from model.nn import update_ema


EPOCHS=20
class TrainLoop:
    def __init__(
        self,
        unet,
        diffusion,
        train_dataLoader,
        lr,
        ema_rate,
        weight_decay=0.0
    ):
        self.unet=unet
        self.diffusion=diffusion
        self.train_dataLoader=train_dataLoader
        self.lr=lr
        self.ema_rate=ema_rate
        self.weight_decay=weight_decay
        self.optimizer = AdamW(self.unet.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(EPOCHS):
            pass
