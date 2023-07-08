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
from tqdm import tqdm




EPOCHS=20
class TrainLoop:
    def __init__(
        self,
        unet,
        diffusion,
        train_dataLoader,
        batch_size,
        lr,
        ema_rate,
        device,
        weight_decay=0.0,
    ):
        self.unet=unet
        self.diffusion=diffusion
        self.train_dataLoader=train_dataLoader
        self.batch_size=batch_size
        self.lr=lr
        self.ema_rate=ema_rate
        self.device=device
        self.weight_decay=weight_decay
        self.optimizer = AdamW(self.unet.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(EPOCHS):
            for batch in tqdm(train_dataLoader):
                print(f"Epoch: {epoch}")
                batch=batch.to(device)
                self.optimizer.zero_grad()
                t=np.random.uniform(0,self.diffusion.num_timesteps,size=(self.batch_size,))
                losses=self.diffusion.training_losses(self.unet,batch,t)
                weights=np.ones([self.batch_size])
                weights = th.from_numpy(weights).float().to(device)
                loss = (losses["loss"] * weights).mean() ## todo
                loss.backward()
                self.optimizer.step()
      
                

            