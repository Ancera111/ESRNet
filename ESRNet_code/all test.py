
from os import listdir
from utils import *
from torch import nn
from models import SRResNet
import time
from PIL import Image
from skimage.metrics import structural_similarity as SSIM
import numpy as np
import cv2 
from tqdm import tqdm

# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 3      # 放大比例

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
# pic_path =  r'F:\szy\old_computer\A.python\pub datasets\Eye OCT Datasets/'
# save_path = r'F:\szy\old_computer\A.python\pub datasets\SRResNet eye OCT 3x/'
# pic_path =  r'F:\szy\old_computer\A.python\pub datasets\OCT2017/'
# save_path = r'F:\szy\old_computer\A.python\pub datasets\SRResNet OCT2017 3x/'
pic_path =  r'F:\szy\old_computer\A.python\OCT\picture\3\youdajpg/'
# save_path = r'F:\szy\old_computer\A.python\OCT\picture\3\youda16conv3x/' # 16conv
save_path = r'F:\szy\old_computer\A.python\OCT\picture\3\youda3x/'       # srresnet


# 预训练模型F:\szy\old_computer\A.python\SRRESNET\SRGAN\results\ori_data 8conv2X.pth
srresnet_checkpoint = r'F:\szy\old_computer\A.python\SRRESNET\SRGAN\results\ori_data3X.pth' 
#srresnet_checkpoint = r'SRRESNET\SRGAN\results\ori_data.pth'
# 加载模型SRResNet 或 SRGAN
checkpoint = torch.load(srresnet_checkpoint,)
srresnet = SRResNet(large_kernel_size=large_kernel_size,
                    small_kernel_size=small_kernel_size,
                    n_channels=n_channels,
                    n_blocks=n_blocks,
                    scaling_factor=scaling_factor)
srresnet = srresnet.to(device)
srresnet.load_state_dict(checkpoint['model'])

srresnet.eval()
model = srresnet
print('Wating......')
# 记录时间
start = time.time()

for each in tqdm(listdir(pic_path),desc='convert LR images to HR images'):
    imgPath = os.path.join(pic_path,each)
 
    # 加载图像
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')
 
    # 双线性上采样
    #Bicubic_img = img.resize((int(img.width // scaling_factor),int(img.height // scaling_factor)),Image.Resampling.BICUBIC)

    #Bicubic_img = img.resize((int(Bicubic_img.width * scaling_factor),int(Bicubic_img.height * scaling_factor)),Image.Resampling.BICUBIC)

    #Bicubic_img.save('/opt/data/szy/OCT/SRRESNET/SRGAN/results/1.22*2.jpg') 

    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)
 
 
    # 转移数据至设备
    lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed
 
    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cuda().detach()  # (1, 3, w*scale, h*scale), in [-1, 1] 
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        #sr_img = img.resize((int(sr_img.width // scaling_factor),int(sr_img.height // scaling_factor)),Image.Resampling.BICUBIC)
        sr_img.save(save_path + each)

i = len(os.listdir(save_path))
print('Average timing {:.3f} second'.format((time.time()-start)/i))
print('%s having %d images'%(save_path,i))

