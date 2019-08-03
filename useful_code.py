打开终端代理
Polipo
/etc/init.d/polipo start

打开vpn
sudo zjuvpn


# -*- coding: utf-8 -*-

#用json文件单独定义参数
with open(path) as f:
    config=json.load(f)
nb_epochs=int(config['epochs'])
batch_size=int(config['batch_size'])
nb_classes=int(config['nb_classes'])

#打乱数据集
train=list(zip(x_train,y_train,train_imgname))
random.shuffle(train)
x_train,y_train,train_imgname=zip(*train)

#搜索文件
mhd_file_list = glob(luna_subset_path + "*.mhd")


from skimage.morphology import label
def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)


#文件夹中没有子文件夹直接是文件情况
for mask_file in next(os.walk(path + '/masks/'))[2]:
    mask_ = cv2.imread(path + '/masks/' + mask_file)
    print(mask_file)
#用map循环运行函数
#它可以将一个函数映射到一个可枚举类型上面
map( lambda x: x*x, [y for y in range(10)] )

#读取csv文件并将id列数据追加字符".png"
def append_ext(fn):
    return fn+".png"

traindf=pd.read_csv(“./trainLabels.csv”,dtype=str)
testdf=pd.read_csv("./sampleSubmission.csv",dtype=str)
traindf["id"]=traindf["id"].apply(append_ext)
testdf["id"]=testdf["id"].apply(append_ext)


#读医学图像数据
#SimpleItk  读任何格式（z，w，h）
ct = sitk.ReadImage(os.path.join(ct_dir, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
#nibabel  读nii格式（w,h,z）
image_data = nib.load(path).get_data()

#example
import cv2
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

path = "/home/zhounan/Documents/dataset/LiverLesionDetection_Splited/0/test/000-2945085-2-0-1/Image_ART.mhd"
image = sitk.ReadImage(path)
image = sitk.GetArrayFromImage(image)

# print(image)
for i in image:
    plt.imshow(i, cmap='gray')
    plt.axis('off')
    plt.show()


#存成npy格式
save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.npy')
                        # print save_path
                        np.save(save_path, np.array(cur_patch))
                        patch_count += 1


#os.walk()
import os
path = "/home/zhounan/Desktop/Resnet"
for dirName, subdirList, fileList in os.walk(path):
    for filename in fileList:
        # print(os.path.join(dirName+filename))
        print(dirName)
        print(filename)
        print('\n')


#文件遍历读取
import os
class_names_to_ids = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
data_dir = '/home/zhounan/PycharmProjects/slim-practice/data/flower_photos/'
output_path = 'list.txt'
fd = open(output_path, 'w')
for class_name in class_names_to_ids.keys():
    images_list = os.listdir(data_dir + class_name)
    for image_name in images_list:
        fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))
fd.close()


#txt格式文件读取
import random
_NUM_VALIDATION = 350
_RANDOM_SEED = 0
list_path = './data/list.txt'
train_list_path = './data/list_train.txt'
val_list_path = './data/list_val.txt'
fd = open(list_path)
lines = fd.readlines()
fd.close()
random.seed(_RANDOM_SEED)
random.shuffle(lines)
fd = open(train_list_path, 'w')
for line in lines[_NUM_VALIDATION:]:
    fd.write(line)
fd.close()
fd = open(val_list_path, 'w')
for line in lines[:_NUM_VALIDATION]:
    fd.write(line)
fd.close()


next(os.walk(path + '/masks/'))[0]


# 调整窗宽 窗位
def rejust_pixel_value(image):
    image = np.array(image)
    ww = np.float64(250)
    wc = np.float64(55)
    ww = max(1, ww)
    lut_min = 0
    lut_max = 255
    lut_range = np.float64(lut_max) - lut_min

    minval = wc - ww / 2.0
    maxval = wc + ww / 2.0
    image[image < minval] = minval
    image[image > maxval] = maxval
    to_scale = (minval <= image) & (image <= maxval)
    image[to_scale] = ((image[to_scale] - minval) / (ww * 1.0)) * lut_range + lut_min
    return image

#打乱给定的数据集顺序
index = [i for i in range(len(data))] 
random.shuffle(index)
data = data[index]
label = label[index]

#给数组加维度（第一维）
img = np.reshape(img, img.shape + (1,))

如果当前文件夹下没有birth_weight.csv文件
if not os.path.exists(birth_weight_file):
    birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
