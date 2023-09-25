import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import skimage
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from ssim import ssim
from tqdm import tqdm
import time
from os import listdir


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])

if __name__ == '__main__':
    cudnn.benchmark = True
    device = torch.device('cuda')

    weights_file = r'F:/szy/old_computer/A.python/SRCNN-pytorch-master/out_out/outputs/srcnn_x4.pth' 
    scale = 4

    Psnr = 0
    Ssim = 0
    start = time.time()
    test_path = 'F:/DL/111111.jpg'
    save_path = 'F:/DL/111112.jpg'
    images_name = [x for x in listdir(test_path) if is_image_file(x)]
    for eachfile in tqdm(images_name,desc = '进度'):
        each_file = test_path + eachfile
        model = SRCNN().to(device)
        state_dict = model.state_dict()
        for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

        model.eval()

        image = pil_image.open(each_file).convert('RGB')

        image_width = (image.width // scale) * scale
        image_height = (image.height // scale) * scale
        image = image.resize((image_width, image_height), resample=pil_image.Resampling.BICUBIC)
        image = image.resize((image.width // scale, image.height // scale), resample=pil_image.Resampling.BICUBIC)
        image = image.resize((image.width * scale, image.height * scale), resample=pil_image.Resampling.BICUBIC)
        #image.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))

        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        y = ycbcr[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)

        psnr = calc_psnr(y, preds)
        ssim1 = ssim(y,preds)
        #print('y:',y)
        #print('preds:',preds)
        #print('PSNR: {:.3f}'.format(psnr))
        #print('SSIM: {:.3f}'.format(ssim1))
        Psnr += psnr 
        Ssim += ssim1
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(save_path + eachfile)

    
    print('AVERAGE PSNR: {:.3f}'.format(Psnr/400))
    print('AVERAGE SSIM: {:.3f}'.format(Ssim/400))
    print('Over!!!')
    #print('Average timing {:.3f} second'.format((time.time()-start)/400))