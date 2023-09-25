#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :create_data_lists.py
@说明        :创建训练集和测试集列表文件
@时间        :2020/02/11 11:32:59
@作者        :钱彬
@版本        :1.0
'''


from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=[r'F:\szy\old_computer\A.python\SRRESNET\SRGAN\HRimg\train'
                                     ],
                      test_folders=[r'F:\szy\old_computer\A.python\SRRESNET\SRGAN\HRimg\val',
                                    #'D:/szy/server/OCT/gray/sr_web',
                                    #'D:/szy/server/OCT/pictures/3/youda SR 4X',
                                    #'D:/szy/server/OCT/pictures/3/youda 8x'
                                    ],    
    # create_data_lists(train_folders=['SRGAN/data/COCO2014/train2014/',
    #                                  'SRGAN/data/COCO2014/val2014/',
    #                                  'SRGAN/data/BSD100/'],
    #                   test_folders=['SRGAN/data/Set5/',
    #                                 'SRGAN/data/Set14/',
    #                                 #'D:/szy/server/OCT/gray/sr_web',
    #                                 #'D:/szy/server/OCT/pictures/3/youda SR 4X',
    #                                 #'D:/szy/server/OCT/pictures/3/youda 8x'
    #                                 ],
                      min_size=100,
                      output_folder=r'F:\szy\old_computer\A.python\SRRESNET\SRGAN\HRimg\output')
