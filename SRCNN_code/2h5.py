from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

##改变图片大小，修改图片名字 ​
def get_smaller(path_in, name, path=None, width=64, length=64):
    ''' 检查文件夹是否建立，并建立文件夹 '''
    if path == None:
        tar = os.path.exists(os.getcwd() + "\\" + name)
        if not tar:
            os.mkdir(os.getcwd() + "\\" + name)
        im_path = os.getcwd() + "\\" + name + "\\"
    else:
        tar = os.path.exists(path + "\\" + name)
        if not tar:
            os.mkdir(path + "\\" + name)
        im_path = path + "\\" + name + "\\"
 
    i = 1
    list_image = os.listdir(path_in)
    for item in list_image:
        '''检查是否有图片'''
        tar = os.path.exists(im_path+str(i)+'.jpg')
        if not tar:
            image = Image.open(path_in+'\\'+item)
            smaller = image.resize((width, length), Image.ANTIALIAS)
            '''注意这里如果不加转换，很可能会有报错'''
            if not smaller.mode == "RGB":
                smaller = smaller.convert('RGB')
            smaller.save(im_path+str(i)+'.jpg')
            i += 1
 
def createData(path):
    pics = os.listdir(path)
    all_data = []
    for item in pics:
        '''难免有图片打不开'''
        try :
            all_data.append(plt.imread(path+'/'+item).tolist())
        except Exception as pic_wrong:
            print(item+" pic wrong")
    return all_data
 
def createSet(hf, name, tip, data):
    hf.create_dataset(name, data=data)
    t = [[tip]*len(data)]
    hf.create_dataset(name + '_tip', data=t)

if __name__ == '__main__':
    hf = h5py.File('data-train.h5', 'w')
    all_data = createData(r'F:\szy\old_computer\A.python\SRRESNET\SRGAN\HRimg\train')
    createSet(hf, 'train_set_1', 1, all_data)
    hf.close()