import sys
import os
cwd = os.getcwd()
sys.path.insert(0,cwd)

from model.Unet import UNetModel
from model import gaussian_diffusion as gd
from dataset.load_dataset import get_train_dataset, get_dataLoader
from training2 import TrainLoop
import torch
import matplotlib.pyplot as plt



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


def main():
    BATCH_SIZE=2
    IMAGE_SIZE=64
    MICROBATCH_SIZE=2
    lr=0.0001
    ema_rate=0.9999
    log_interval=10
    save_interval=10000
    use_fp16=False
    fp16_scale_growth=0.001
    weight_decay=0.0
    lr_anneal_steps=0
    resume_checkpoint=""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    unet=create_Unet_model(
        image_size=IMAGE_SIZE,
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
        dropout=0.0,

    )
    diffusion=create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        timestep_respacing=False,)
    
    train_dataset=get_train_dataset("MVTecAD/carpet/train/good/",image_size=IMAGE_SIZE)
    # print(train_dataset.__getitem__(0))
    # from torchvision import datasets, transforms, models
    # a=train_dataset.__getitem__(0)
    # print(a.max())
    # print(a.min())
    # b=(a+1)*127.5
    # print(b.max())
    # print(b.min())
    # plt.imsave("image.jpg",transforms.ToPILImage()(a*255))
    # exit(0)
    train_data_loader=get_dataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    unet=unet.to(device=device)
    TrainLoop(
        unet=unet,
        diffusion=diffusion,
        train_dataLoader=train_data_loader,
        batch_size=BATCH_SIZE,
        lr=lr,
        ema_rate=ema_rate,
        num_epochs=1000,
        base_model_path="../drive/MyDrive/FreeRAD/models/",
        device=device,
        weight_decay=weight_decay,
        resume_training=False,
        last_checkpoint=-1,
    ).run_loop()



if __name__ == "__main__":
    main()
