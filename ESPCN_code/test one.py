import argparse
import os
from os import listdir
import time
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm
from model import Net

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])

if __name__ == "__main__":

    upscale_factor = 2
    model_name= 'epoch_2_10.pt'

    UPSCALE_FACTOR = upscale_factor
    MODEL_NAME = model_name


    image_name = '003L.png'

    model = Net(upscale_factor=UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('ESPCN/ESPCN-master/epochs/' + MODEL_NAME))
    

    start = time.time()


    img = Image.open(image_name).convert('YCbCr')
    y, cb, cr = img.split()
    image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    if torch.cuda.is_available():
        image = image.cuda()

    out = model(image)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    out_img.save(image_name.replace('.', '_espcn_x{}.'.format(upscale_factor)))

    print('Over!')
    #lenth = len(os.listdir(out_path))
    #print(lenth)
    #print('Average timing {:.3f} second'.format((time.time()-start)/lenth))