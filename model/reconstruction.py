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
from tqdm import tqdm
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
        self.IMAGE_SIZE = 256
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
        self.model=self.model.to(self.device)
        if model_path is not None:
            self.load_model(model_path,self.device)
            
    def load_model(self, path, device):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model=self.model.to(self.device)
        self.model.eval()


    def one_shot_reconstruct(self, x, t):
      with torch.no_grad():            
        # print(x.shape)
        # print(x.max())
        # print(x.min())
        # print(t.shape)
        # print(t)
        # print("-"*500)
        noisy=self.gd.q_sample(x,t)
        reconstructed=self.gd.p_sample(self.model,noisy,t)
        return reconstructed

    def one_shot_reconstruct_with_noisy(self, x, t):
      with torch.no_grad():            
        # print(x.shape)
        # print(x.max())
        # print(x.min())
        # print(t.shape)
        # print(t)
        # print("-"*500)
        noisy=self.gd.q_sample(x,t)
        reconstructed=self.gd.p_sample(self.model,noisy,t)
        return reconstructed,noisy


    
    def calc_error_map(self,images,reconstructed_images,scale):
        resized_images=resize_image(images,scale*images.shape[2],scale*images.shape[3])
        resized_reconstructed=resize_image(reconstructed_images,scale*reconstructed_images.shape[2],scale*reconstructed_images.shape[3])
        error_mp=(resized_images-resized_reconstructed)**2
        error_mp=torch.mean(error_mp,dim=1)
        error_mp = torch.unsqueeze(error_mp, dim=1)
        error_mp=resize_image(error_mp,(1/scale)*error_mp.shape[2],(1/scale)*error_mp.shape[3])
        error_mp = torch.squeeze(error_mp, dim=1)
        return error_mp
        

    def calc_error_ms(self,images,reconstructed_images,filter_size=3):
        import torch.nn.functional as F
        scales=[1,0.5,0.25,0.125]
        error_maps = torch.empty((len(scales), self.IMAGE_SIZE, self.IMAGE_SIZE),device=self.device)
        for i,scale in enumerate(scales):
            error_mp=self.calc_error_map(images.clone(),reconstructed_images.clone(),scale)
            error_maps[i]=error_mp
        mean_err_map=torch.mean(error_maps,dim=0).to(self.device) 
        mean_err_map = torch.unsqueeze(mean_err_map, dim=0)
        mean_kernel = torch.ones((1,1,filter_size, filter_size)) / (filter_size * filter_size)
        mean_kernel=mean_kernel.to(self.device)
        mean_err_map = F.conv2d(mean_err_map, mean_kernel, padding=filter_size // 2)
        return mean_err_map

    def calc_error_ms_of_training_data(self,train_loader,t):
        training_error_ms = torch.empty((len(train_loader), self.IMAGE_SIZE, self.IMAGE_SIZE),device=self.device)
        for i,train_batch in enumerate(train_loader):
            train_batch=train_batch.to(self.device)
            reconstructed_images=self.one_shot_reconstruct(train_batch,t)["pred_xstart"]
            training_error_ms[i]=self.calc_error_ms(train_batch,reconstructed_images)
        mean_training_error_ms=torch.mean(training_error_ms,dim=0)
        return mean_training_error_ms


    def anomaly_score_calculation(self,images,reconstructed_images,mean_training_error_ms=None,mean_filter_size=3):
        error_ms=self.calc_error_ms(images,reconstructed_images,mean_filter_size)
        anomaly_score=torch.max(torch.abs(error_ms-mean_training_error_ms).view(error_ms.shape[0],-1),dim=1)[0]
        return anomaly_score.tolist()

    def myAnomalyScore(self,image,reconstructed):
        image_arr=image.detach().cpu().numpy()
        reconstructed_arr=reconstructed.detach().cpu().numpy()
        image_arr=image_arr.astype("float64")
        reconstructed_arr=reconstructed_arr.astype("float64")
        err=(image_arr-reconstructed_arr)**2
        sum_err=np.sum(err)
        print(sum_err)
        return sum_err


    def save_result(self,scores,labels,result_name):
        labels=np.array(labels)
        scores=np.array(scores)
        normal_scores=scores[np.where(labels==0)]
        anomaly_scores=scores[np.where(labels==1)]
        x = list(range(1,len(normal_scores)+1,1))
        y = list(range(1,len(anomaly_scores)+1,1))
        plt.scatter(x, normal_scores, label= "normal reconstrcution error", color= "blue", marker= "*", s=30)
        plt.scatter(y, anomaly_scores, label= "anomaly reconstrcution error", color= "red", marker= "*", s=30)
        plt.xlabel('Data points')
        plt.ylabel('reconstruction error')
        plt.title('Reconstruction error distribution')
        plt.legend()
        plt.savefig(f'{result_name}.png')
        plt.clf()
        plt.close()


def plot_images(image1_list,image2_list,diffused_list,image3_list,result_name):
    import cv2 as cv
    f, axs = plt.subplots(len(image1_list),4,figsize=(20,160))
    for i in range(len(image1_list)):
        # image3=np.abs(image1_list[i]-image2_list[i])
        # image3=cv.cvtColor(image3_list[i],cv.COLOR_RGB2GRAY)
        # cv.imwrite("gray.jpg",image3)
        image3=image3_list[i]
        axs[i,0].imshow(image1_list[i][:,:,::-1])
        axs[i,1].imshow(diffused_list[i][:,:,::-1])
        axs[i,2].imshow(image2_list[i][:,:,::-1])
        print(image3.shape)
        print(f"---------{np.max(image3)},{np.min(image3)}")
        axs[i,3].imshow(image3,cmap="gray")
    plt.savefig(f'{result_name}.png')
    plt.clf()
    plt.close()



    
def main():
    BATCH_SIZE=1
    IMAGE_SIZE=256
    path="MVTecAD/hazelnut/test/print/"
    entries = os.listdir(path=path)
    print_cat=[path+image_name for image_name in entries]
    print_cat_label=[1 for i in range(len(print_cat))]
    
    path="MVTecAD/hazelnut/test/crack/"
    entries = os.listdir(path=path)
    crack=[path+image_name for image_name in entries]
    crack_label=[1 for i in range(len(crack))]
    
    
    path="MVTecAD/hazelnut/test/good/"
    entries = os.listdir(path=path)
    good=[path+image_name for image_name in entries]
    good_label=[0 for i in range(len(good))]
    
    image_paths=print_cat+crack+good
    labels=print_cat_label+crack_label+good_label
    test_dataset=get_test_dataset(image_paths=image_paths,labels=labels,image_size=IMAGE_SIZE)
    test_data_loader=get_dataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    # original_test_dataset=get_test_dataset(image_paths=image_paths,labels=labels,image_size=IMAGE_SIZE,transform=False)
    # original_test_data_loader=get_dataLoader(original_test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    reconstructor=Reconstructor("../drive/MyDrive/FreeRAD/models/HazelnutCheckpoint_ep77.pt",device=device)
    
    train_dataset=get_train_dataset("MVTecAD/hazelnut/train/good/",image_size=IMAGE_SIZE)
    train_loader=get_dataLoader(train_dataset,BATCH_SIZE,False)
    t=np.array([200 for i in range(BATCH_SIZE)])
    t = torch.from_numpy(t).long().to(device)            
    mean_error_maps_of_traing=reconstructor.calc_error_ms_of_training_data(train_loader,t)
    anomaly_scores=[]
    anomaly_labels=[]
    my_anomaly_score=[]
    image1_list=[]
    image2_list=[]
    image3_list=[]
    diffused_list=[]
    for i,batch in enumerate(tqdm(test_data_loader)):
                images,batch_labels=batch
                images=images.to(device)
                batch_labels=batch_labels.to(device)
                t=np.array([200 for i in range(len(batch_labels))])
                t = torch.from_numpy(t).long().to(device)
                reconstructed_images,diffused_image=reconstructor.one_shot_reconstruct_with_noisy(images,t)
                reconstructed_images=reconstructed_images["pred_xstart"]
                score=reconstructor.anomaly_score_calculation(images,reconstructed_images,mean_error_maps_of_traing)
                error_ms=reconstructor.calc_error_ms(images,reconstructed_images)
                error_map=torch.abs(error_ms-mean_error_maps_of_traing)
                error_map=(error_map+1)*0.5*255
                error_map[error_map>255]=255
                error_map[error_map<0]=0
                error_map=error_map.detach().cpu().numpy().reshape(256,256).astype("uint8")
                image3_list.append(error_map)
                my_score=[reconstructor.myAnomalyScore(images,reconstructed_images)]
                image1=(images+1)*0.5*255
                image1[image1>255]=255
                image1[image1<0]=0
                image1=image1.detach().cpu().numpy()[0,:,:,:].reshape(256,256,3).astype("uint8")
                image2=(reconstructed_images+1)*0.5*255
                image2[image2>255]=255
                image2[image2<0]=0
                image2=image2.detach().cpu().numpy()[0,:,:,:].reshape(256,256,3).astype("uint8")
                
                diffused=(diffused_image+1)*0.5*255
                diffused[diffused>255]=255
                diffused[diffused<0]=0
                diffused=diffused.detach().cpu().numpy()[0,:,:,:].reshape(256,256,3).astype("uint8")
                diffused_list.append(diffused)
                image1_list.append(image1)
                image2_list.append(image2)
                # plot_images(image1,image2,f"result_{i}")
                anomaly_scores=anomaly_scores+score
                my_anomaly_score=my_anomaly_score+my_score
                anomaly_labels=anomaly_labels+batch_labels.tolist()
    print(anomaly_scores)
    print(anomaly_labels)

    plot_images(image1_list[:len(print_cat)],
                image2_list[:len(print_cat)],
                diffused_list[:len(print_cat)],
                image3_list[:len(print_cat)],
                "PrintHazelnut")
    
    plot_images(image1_list[len(print_cat):len(print_cat)+len(crack)],
                image2_list[len(print_cat):len(print_cat)+len(crack)],
                diffused_list[len(print_cat):len(print_cat)+len(crack)],
                image3_list[len(print_cat):len(print_cat)+len(crack)],
                "CrackHazelnut")
    
    plot_images(image1_list[len(print_cat)+len(crack):len(print_cat)+len(crack)+len(good)],
                image2_list[len(print_cat)+len(crack):len(print_cat)+len(crack)+len(good)],
                diffused_list[len(print_cat)+len(crack):len(print_cat)+len(crack)+len(good)],
                image3_list[len(print_cat)+len(crack):len(print_cat)+len(crack)+len(good)],
                "GoodHazelnut")
    
    reconstructor.save_result(anomaly_scores,anomaly_labels,"Reconstruction_error")
    reconstructor.save_result(my_anomaly_score,anomaly_labels,"MyReconstruction_error")
            


if __name__ == "__main__":
    main()
