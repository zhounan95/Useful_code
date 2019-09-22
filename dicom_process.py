# -*- coding: utf-8 -*- 
'''
https://blog.csdn.net/JianJuly/article/details/81214408
https://zhidao.baidu.com/question/140557565574560405.html
读取dicom头文件中所有信息
'''

import pydicom

path1 = '/home/zhounan/Documents/CT1081358/30044.dcm'
ds = pydicom.read_file(path1)
for i in ds:
    i = str(i)
    # print(i)
    if i.find('Series Number') > 0:
        print(i)
        print(i.split("\"")[1])