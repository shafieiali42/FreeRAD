from model.Unet import SuperResModel, UNetModel
from model import gaussian_diffusion as gd




def main():
    unet=create_Unet_model(
        image_size=64,
        num_channels=128,
        num_res_blocks=3,
        learn_sigma=True,
        class_cond=False,
        NUM_CLASSES=1000,    
        use_checkpoint=False,
        attention_resolutions="16,8",
        num_heads=4,
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
    
    schedule_sampler = None
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )


    

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
