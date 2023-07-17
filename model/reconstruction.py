import sys
import os

cwd = os.getcwd()
sys.path.insert(0, cwd)

from model.Unet import UNetModel
from model import gaussian_diffusion as gd
from dataset.load_dataset import get_test_dataset, get_dataLoader
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
        reconstructed=self.gd.p_sample(self.model,noisy,t)
        return reconstructed

    def calc_mean_error_maps_of_traing():
        pass


    def resize_batch_by_size(img_batch, resize_width, resize_height):
        """
        :params
            image: np.array(), shape -> (batch, width, height, channels)
            resize_width: The resize width dimension. 
            resize_height: The resize height dimension. 

        :returns
            array of shape -> (batch, resized_width, resized_height, channels)
        """
        batch, original_width, original_height, channel = img_batch.shape
        
        rd_ch = img_batch[:,:,:,0]
        gr_ch = img_batch[:,:,:,1]
        bl_ch = img_batch[:,:,:,2]
        
        resized_images = np.zeros((batch, resize_width, resize_height, channel), dtype=np.uint8)
        
        x_scale = original_width/resize_width
        y_scale = original_height/resize_height
        
        resize_idx = np.zeros((resize_width, resize_height))
        resize_index_x = np.ceil(np.arange(0, original_width, x_scale)).astype(int)
        resize_index_y = np.ceil(np.arange(0, original_height, y_scale)).astype(int)
        resize_index_x[np.where(resize_index_x == original_width)]  -= 1
        resize_index_y[np.where(resize_index_y == original_height)] -= 1
        
        resized_images[:,:,:,0] = rd_ch[:,resize_index_x,:][:,:,resize_index_y]
        resized_images[:,:,:,1] = gr_ch[:,resize_index_x,:][:,:,resize_index_y]
        resized_images[:,:,:,2] = bl_ch[:,resize_index_x,:][:,:,resize_index_y]
        
        return resized_images
    def resize_batch_by_scale(self,img_batch, fx, fy):
        """
        :params
            image: np.array(), shape -> (batch, width, height, channels)
            resize_width: The resize width dimension. 
            resize_height: The resize height dimension. 

        :returns
            array of shape -> (batch, resized_width, resized_height, channels)
        """
        
        batch, original_width, original_height, channel = img_batch.shape
        resize_width=int(original_width*fy)
        resize_height=int(original_height*fx)

        rd_ch = img_batch[:,:,:,0]
        gr_ch = img_batch[:,:,:,1]
        bl_ch = img_batch[:,:,:,2]
        
        resized_images = np.zeros((batch, resize_width, resize_height, channel), dtype=np.uint8)
        
        
        resize_idx = np.zeros((resize_width, resize_height))
        resize_index_x = np.ceil(np.arange(0, original_width, fx)).astype(int)
        resize_index_y = np.ceil(np.arange(0, original_height, fy)).astype(int)
        resize_index_x[np.where(resize_index_x == original_width)]  -= 1
        resize_index_y[np.where(resize_index_y == original_height)] -= 1
        
        resized_images[:,:,:,0] = rd_ch[:,resize_index_x,:][:,:,resize_index_y]
        resized_images[:,:,:,1] = gr_ch[:,resize_index_x,:][:,:,resize_index_y]
        resized_images[:,:,:,2] = bl_ch[:,resize_index_x,:][:,:,resize_index_y]
        
        return resized_images
    
    def anomaly_score_calculation(self,images,reconstructed_images,mean_error_maps_of_traing=None,chanel_axis=2,mean_filter_size=3):
        images_copy=images.copy()
        reconstructed_images_copy=reconstructed_images.copy()
        images_copy = images_copy.astype("float64")
        reconstructed_images_copy = reconstructed_images_copy.astype("float64")
        scales=[1,0.5,0.25,0.125]
        error_maps=[]
        for scale in scales:
            resized_image = self.resize_batch_by_scale(images_copy,scale,scale)
            resized_reconstructed = self.resize_batch_by_scale(reconstructed_images_copy,scale,scale)
            diff=(resized_image-resized_reconstructed)**2
            err_l=np.mean(diff,axis=chanel_axis)
            err_l=self.resize_batch_by_size(err_l,images_copy.shape[2],images_copy.shape[1])
            error_maps.append(err_l)
        mean_err_map=np.zeros_like(error_maps[0])
        for i in range(len(error_maps)):
            mean_err_map=mean_err_map+error_maps[i]/len(error_maps)
        filter=np.ones((mean_filter_size,mean_filter_size),dtype="float64")/(mean_filter_size**2)
        mean_err_map=cv.filter2D(mean_err_map,-1,filter,borderType=cv.BORDER_CONSTANT)

        plt.imshow(mean_err_map,cmap="gray")


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
    mean_error_maps_of_traing=reconstructor.calc_mean_error_maps_of_traing()
    for batch in tqdm(test_data_loader):
                images,batch_labels=batch
                images=images.to(device)
                batch_labels=batch_labels.to(device)
                t=np.array([20 for i in range(len(labels))])
                t = torch.from_numpy(t).long().to(device)
                reconstructed_images=reconstructor.one_shot_reconstruct(images,t)
                reconstructor.anomaly_score_calculation(images,reconstructed_images,mean_error_maps_of_traing=mean_error_maps_of_traing,chanel_axis=)
                
            
    

if __name__ == "__main__":
    main()
