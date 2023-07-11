import sys
import os

cwd = os.getcwd()
sys.path.insert(0, cwd)

from model.Unet import UNetModel
from model import gaussian_diffusion as gd
from dataset.load_dataset import get_train_dataset, get_dataLoader
import torch
import matplotlib.pyplot as plt
from PIL import Image



def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return gd.GaussianDiffusion(
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.LEARNED_RANGE,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def create_Unet_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma,
        class_cond,
        NUM_CLASSES,
        use_checkpoint,
        attention_resolutions,
        num_heads,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


class Reconstructor:
    def __init__(self, model_path, device='cpu'):
        self.IMAGE_SIZE = 64
        self.gd =create_gaussian_diffusion(
                    steps=1000,
                    learn_sigma=True,
                    sigma_small=False,
                    noise_schedule="linear",
                    use_kl=False,
                    predict_xstart=False,
                    rescale_timesteps=True,
                    rescale_learned_sigmas=True,
                    timestep_respacing=False,)
                
        self.model = create_Unet_model(image_size=self.IMAGE_SIZE,
                                       num_channels=128,
                                       num_res_blocks=3,
                                       learn_sigma=True,
                                       class_cond=False,
                                       NUM_CLASSES=1000,
                                       use_checkpoint=False,
                                       attention_resolutions="16,16",
                                       num_heads=1,
                                       num_heads_upsample=-1,
                                       use_scale_shift_norm=True,
                                       dropout=0.0, )
        if model_path is not None:
            self.load_model(model_path, device)

    def load_model(self, path, device="cpu"):
        checkpoint = torch.load(path,map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def one_shot_reconstruct(self, x, t):
        noisy=self.gd.q_sample(x,t)
        model_output = self.model(x, self.gd._scale_timesteps(t))
        model_output, model_var_values = torch.split(model_output, 3, dim=1)
        reconstructed=self.gd._predict_xstart_from_eps(noisy,t,model_output)
        # reconstructed=self.gd.p_sample(self.model,noisy,t)
        return reconstructed




def main():
    from torchvision import datasets, transforms, models
    from dataset import load_dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reconstructor=Reconstructor("../drive/MyDrive/FreeRAD/models/checkpoint_ep_final215.pt",device=device)
    # reconstructor=Reconstructor(model_path=None,device=device)
    good_image=Image.open("MVTecAD/carpet/train/good/000.png")    
    bad_image=Image.open("MVTecAD/carpet/test/hole/000.png") 
    my_transforms=load_dataset.get_my_transforms(image_size=reconstructor.IMAGE_SIZE)
    good_image_transformed=my_transforms(good_image)
    bad_image_transformed=my_transforms(bad_image)
    # print(good_image_transformed)
    good_image_reconstructed=reconstructor.one_shot_reconstruct(good_image_transformed,20)
    bad_image_reconstructed=reconstructor.one_shot_reconstruct(bad_image_transformed,20)


if __name__ == "__main__":
    main()
