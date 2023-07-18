import numpy as np
import torch
from torchvision import transforms

def resize_image(images,new_height, new_width):
  new_height=int(new_height)
  new_width=int(new_width)
  batch_size, channels, height, width = images.size()
  reshaped_images = images.view(batch_size * channels, height, width)
  resize_transform = transforms.Compose([
      transforms.ToPILImage(), 
      transforms.Resize((new_height, new_width)),  
      transforms.ToTensor() 
  ])
  resized_images = torch.stack([resize_transform(img) for img in reshaped_images])
  resized_images = resized_images.view(batch_size, channels, new_height, new_width)
  return resized_images

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
def resize_batch_by_scale(img_batch, fx, fy):
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
