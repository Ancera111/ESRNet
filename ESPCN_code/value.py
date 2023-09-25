from calc import calc_psnr,ssim
import numpy as np
from PIL import Image
import torch
import cv2 as cv
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Psnr = 0
Ssim = 0
for i in range(400):

    image1_path = "data/mydata/train/%d.png"%(i+1)
    image1 = Image.open(image1_path)
    img_arr = np.array(image1)
    trans = transforms.ToTensor() #转为tensor格式
    tensor_img1 = trans(image1).reshape([1,3, 248, 345]) #转为4D 才能算ssim
    image2_path = "data/mydata/result4/%d.png"%(i+1)
    image2 = Image.open(image2_path).resize((345,248))  #更改原图大小
    img_arr = np.array(image2)
    trans = transforms.ToTensor()
    tensor_img2 = trans(image2).reshape([1,3, 248, 345])

    psnr = calc_psnr(tensor_img1, tensor_img2)
    Psnr += float(psnr)
    ssim1 = ssim(tensor_img1, tensor_img2)
    Ssim += float(ssim1)

print('4x Average PSNR: {:.3f}'.format(Psnr/400))
print('4x Average SSIM: {:.3f}'.format(Ssim/400))