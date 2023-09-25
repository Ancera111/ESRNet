import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from ssim import ssim
from models import FSRCNN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
import time
from tqdm import tqdm
from os import listdir

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    weights_file = r'F:/szy/old_computer/A.python/FSRCNN-pytorch-master/BLAH_BLAH/fsrcnn_x4.pth' 
    scale = 4

    start = time.time()
    test_path = 'FSRCNN-pytorch-master/train/'
    save_path = 'FSRCNN-pytorch-master/result4/'


    images_name = [x for x in listdir(test_path) if is_image_file(x)]
    for eachfile in tqdm(images_name,desc = '进度'):
        image_file = test_path + eachfile
        

        cudnn.benchmark = True

        model = FSRCNN(scale_factor=scale).to(device)

        state_dict = model.state_dict()
        for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

        model.eval()
        
        image = pil_image.open(image_file).convert('RGB')

        image_width = (image.width // scale) * scale
        image_height = (image.height // scale) * scale

        hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // scale, hr.height // scale), resample=pil_image.BICUBIC)
        bicubic = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
        #bicubic.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))

        lr, _ = preprocess(lr, device)
        hr, _ = preprocess(hr, device)
        _, ycbcr = preprocess(bicubic, device)

        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)

        psnr = calc_psnr(hr, preds)
        ssim1 = ssim(hr,preds)
        #print('y:',y)
        #print('preds:',preds)
        #print('PSNR: {:.3f}'.format(psnr))
        #print('SSIM: {:.3f}'.format(ssim1))

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(save_path + eachfile)


    
    print('Average timing {:.3f} second'.format((time.time()-start)/400))