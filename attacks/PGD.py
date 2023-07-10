import torchattacks
import torch


class PGD:
    def __init__(self,model:torch.nn.Module,
                 eps=8.0/255,
                 alpha=2.0/255,
                 ):
        self.model=model
        self.eps=eps
        self.alpha=alpha
        self.attacker=torchattacks.PGD(model,eps, alpha)


    def generate_adversarial_images(self,x,y):
        return self.attacker(x,y)




