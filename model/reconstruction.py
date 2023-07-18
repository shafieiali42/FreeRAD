import sys
import os

cwd = os.getcwd()
sys.path.insert(0, cwd)

from model.Unet import UNetModel
from model import gaussian_diffusion as gd
from dataset.load_dataset import get_test_dataset,get_train_dataset, get_dataLoader
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import numpy as np
from torchvision import datasets, transforms, models
from dataset import load_dataset
import torch
import matplotlib.pyplot as plt
import tqdm
from utils.my_utils import resize_batch_by_scale,resize_batch_by_size,resize_image



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
        self.device=device
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
        self.model=self.model.to(device)
        if model_path is not None:
            self.load_model(model_path, device)

    def load_model(self, path, device="cpu"):
        checkpoint = torch.load(path,map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def one_shot_reconstruct(self, x, t):
        noisy=self.gd.q_sample(x,t)
        reconstructed=self.gd.p_sample(self.model,noisy,t)
        return reconstructed

    

    
    def calc_error_map(self,images,reconstructed_images,scale):
        resized_images=resize_image(images,scale*images.shape[2],scale*images.shape[3])
        resized_reconstructed=resize_image(reconstructed_images,scale*reconstructed_images.shape[2],scale*reconstructed_images.shape[3])
        error_mp=(resized_images-resized_reconstructed)**2
        error_mp=torch.mean(error_mp,dim=1)
        error_mp = torch.unsqueeze(error_mp, dim=1)
        error_mp=resize_image(error_mp,scale*error_mp.shape[2],scale*error_mp.shape[3])
        error_mp = torch.squeeze(error_mp, dim=1)
        return error_mp
        

    def calc_error_ms(self,images,reconstructed_images,filter_size=3):
        import torch.nn.functional as F
        scales=[1,0.5,0.25,0.125]
        error_maps=[]
        for scale in scales:
            error_mp=self.calc_error_map(images,reconstructed_images,scale)
            error_maps.append(error_mp)
        mean_err_map=torch.zeros_like(error_maps[0],device=self.device)
        for i in range(len(error_maps)):
            mean_err_map=mean_err_map+error_maps[i]/len(error_maps)
        
        mean_err_map = torch.unsqueeze(mean_err_map, dim=1)
        mean_kernel = torch.ones((1, 1, filter_size, filter_size)) / (filter_size * filter_size)
        mean_kernel=mean_kernel.to(self.device)
        mean_err_map = F.conv2d(mean_err_map, mean_kernel, padding=filter_size // 2)
        mean_err_map=torch.squeeze(mean_err_map,dim=1)
        return mean_err_map

    def calc_error_ms_of_training_data(self,train_loader,t):
        training_error_ms = torch.empty((len(train_loader), self.IMAGE_SIZE, self.IMAGE_SIZE))
        for i,train_batch in enumerate(train_loader):
            train_batch=train_batch.to(self.device)
            reconstructed_images=self.one_shot_reconstruct(train_batch,t)
            training_error_ms[i]=self.calc_error_ms(train_batch,reconstructed_images)
        mean_training_error_ms=torch.mean(training_error_ms,dim=0)
        return mean_training_error_ms


    def anomaly_score_calculation(self,images,reconstructed_images,mean_training_error_ms=None,mean_filter_size=3):
        error_ms=self.calc_error_ms(images,reconstructed_images,mean_filter_size)
        anomaly_score=torch.max(torch.abs(error_ms-mean_training_error_ms).view(error_ms.shape[0],-1),dim=1)[0]
        return anomaly_score.tolist()



def main():
    BATCH_SIZE=1
    IMAGE_SIZE=64
    path="MVTecAD/carpet/test/metal_contamination"
    entries = os.listdir(path=path)
    contamination=[path+image_name for image_name in entries]
    contamination_label=[1 for i in range(len(contamination))]
    
    path="MVTecAD/carpet/test/cut"
    entries = os.listdir(path=path)
    cut=[path+image_name for image_name in entries]
    cut_label=[1 for i in range(len(cut))]
    
    
    path="MVTecAD/carpet/test/good"
    entries = os.listdir(path=path)
    good=[path+image_name for image_name in entries]
    good_label=[1 for i in range(len(good))]
    
    image_paths=contamination+cut+good
    labels=contamination_label+cut_label+good_label
    test_dataset=get_test_dataset(image_paths=image_paths,labels=labels,image_size=IMAGE_SIZE)
    test_data_loader=get_dataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    reconstructor=Reconstructor("../drive/MyDrive/FreeRAD/models/checkpoint_ep_final215.pt",device=device)
    reconstructor=Reconstructor(model_path=None,device=device)

    train_dataset=get_train_dataset("MVTecAD/carpet/train/good/",image_size=IMAGE_SIZE)
    train_loader=get_dataLoader(train_dataset,BATCH_SIZE,False)
    t=np.array([20 for i in range(BATCH_SIZE)])
    t = torch.from_numpy(t).long().to(device)            
    mean_error_maps_of_traing=reconstructor.calc_error_ms_of_training_data(train_dataset,t)
    anomaly_scores=[]
    for batch in tqdm(test_data_loader):
                images,batch_labels=batch
                images=images.to(device)
                batch_labels=batch_labels.to(device)
                t=np.array([20 for i in range(len(labels))])
                t = torch.from_numpy(t).long().to(device)
                reconstructed_images=reconstructor.one_shot_reconstruct(images,t)
                score=reconstructor.anomaly_score_calculation(images,reconstructed_images,mean_error_maps_of_traing)
                anomaly_scores=anomaly_scores+score
    print(anomaly_scores)
            
    

if __name__ == "__main__":
    main()
