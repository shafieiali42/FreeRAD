import copy
import functools
import os
import numpy as np
import torch as th
import torch.distributed as dist
from torch.optim import AdamW
from model.ema import EMA
from model.nn import update_ema
from tqdm import tqdm




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
        num_epochs,
        base_model_path,
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
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.unet).eval().requires_grad_(False)
        self.optimizer = AdamW(self.unet.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.num_epochs=num_epochs
        self.base_model_path=base_model_path

    def run_loop(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch: {epoch}/{self.num_epochs}")
            for batch in tqdm(self.train_dataLoader):
                batch=batch.to(self.device)
                self.optimizer.zero_grad()
                t=np.random.uniform(0,self.diffusion.num_timesteps,size=(self.batch_size,))
                t = th.from_numpy(t).long().to(self.device)
                losses=self.diffusion.training_losses(self.unet,batch,t)
                weights=np.ones([self.batch_size])
                weights = th.from_numpy(weights).float().to(self.device)
                loss = (losses["loss"] * weights).mean() ## todo
                loss.backward()
                print(losses)
                print("-"*500)
                print(loss)
                self.optimizer.step()
                self.ema.step_ema(self.ema_model, self.unet)
            th.save({'epoch': epoch,'model_state_dict': self.unet.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),}, self.base_model_path+f'checkpoint_ep{epoch}.pt')
