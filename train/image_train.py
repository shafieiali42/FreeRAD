from model.Unet import UNetModel
from model import gaussian_diffusion as gd
from dataset.load_dataset import get_train_dataset, get_dataLoader
from training2 import TrainLoop



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
    BATCH_SIZE=128
    MICROBATCH_SIZE=2
    lr=0.0001
    ema_rate=0.9999
    schedule_sampler = None
    log_interval=10
    save_interval=10000
    use_fp16=False
    fp16_scale_growth=0.001
    weight_decay=0.0
    lr_anneal_steps=0
    resume_checkpoint=""

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
        steps=1,
        learn_sigma=True,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        timestep_respacing=False,)
    
    train_dataset=get_train_dataset("carpet/train/good/",64)
    train_data_loader=get_dataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

    TrainLoop(
        unet=unet,
        diffusion=diffusion,
        data=train_data_loader,
        batch_size=BATCH_SIZE,
        microbatch=MICROBATCH_SIZE,
        lr=lr,
        ema_rate=ema_rate,
        log_interval=log_interval,
        save_interval=save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=use_fp16,
        fp16_scale_growth=fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=weight_decay,
        lr_anneal_steps=lr_anneal_steps,
    ).run_loop()



if __name__ == "__main__":
    main()
