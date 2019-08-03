import cv2
import os
import pydicom
import SimpleITK
import shutil

import os
# import radiomics
# import logging
# import six
import numpy as np
from numpy import *
# import radiomics
# from radiomics import featureextractor
import SimpleITK as sitk
# from radiomics.imageoperations import checkMask
import time
import scipy.io as sio
from Tools import MHD
from sklearn import preprocessing

from sklearn import svm
from sklearn.linear_model import RidgeClassifier, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import naive_bayes
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_predict,cross_val_score,train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import SimpleITK as itk

from front_end import models
from django.db.models import Q
from sklearn.cluster import KMeans

def rename(PathDicom):
    for dirName, subdirList, fileList in os.walk(PathDicom):
        if (len(fileList) > 20):
            # rename部分
            for filename in fileList:
                if(filename != '.DS_Store'):
                    pre = filename.split('_')[0]
                    count = filename.split('_')[1]
                    if(len(count)==1):
                        count = "00"+count
                    elif(len(count)==2):
                        count = "0" + count
                    os.rename(os.path.join(dirName, filename),os.path.join(dirName,pre +'_'+ str(count)))

def DICOMtoMHD(PathDicom,datasetName):
    dataset = models.Dataset.objects.get(Q(datasetName=datasetName))
    numOfCases = -1 #.DS_Store
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for sub in subdirList:
            if sub == 'PV':
                for filename in fileList:
                    if not filename.endswith('.DS_Store'):
                        labeltxt = os.path.join(dirName,filename)
                        f = open(labeltxt, 'r')
                        a = f.read()
                        Dict = eval(a)
                        tumorType = Dict['tumorType']
            if sub == 'mask':
                for maskfile in os.listdir(os.path.join(dirName,sub)):
                    patientName = dirName.split('/')[-2]
                    phase = dirName.split('/')[-1]
                    suffix = maskfile.split('.')[-1]
                    MHDdir = PathDicom + '/unzip/MHD/' + patientName
                    MHDName = phase + '_mask.' + suffix
                    if not os.path.exists(MHDdir):
                        os.makedirs(MHDdir)
                    # shutil.copy(os.path.join(dirName,sub,maskfile), os.path.join(MHDdir,MHDName))

                    shutil.copy(os.path.join(dirName, sub, maskfile), MHDdir)
                    # os.rename(os.path.join(MHDdir,maskfile),os.path.join(MHDdir,MHDName))
                    # for filename in os.listdir(MHDdir):
                    #     os.rename(os.path.join(MHDdir, filename), os.path.join(MHDdir,MHDName))
            if sub == 'DICOM' and '__MACOSX' not in dirName:
                numOfCases = len(os.listdir(os.path.join(dirName,sub))) - 1

        if (len(fileList) > 10 and '__MACOSX' not in dirName):
            patientName = dirName.split('/')[-3]
            phase = dirName.split('/')[-2]
            type = dirName.split('/')[-1].split('_')[0]
            MHDdir = PathDicom + '/unzip/MHD/' + patientName
            MHDName = phase + '_' + type + '.mhd'
            if not os.path.exists(MHDdir):
                os.makedirs(MHDdir)

            lstFilesDCM = []

            for filename in fileList:
                if (filename != '.DS_Store'):
                    lstFilesDCM.append(os.path.join(dirName, filename))  # 加入到列表中
            lstFilesDCM.sort()
            lstFilesDCM.reverse()

            # 第一步：将第一张图片作为参考图片，并认为所有图片具有相同维度
            RefDs = pydicom.read_file(lstFilesDCM[0])  # 读取第一张dicom图片

            patientAge = RefDs.PatientAge
            patientSex = RefDs.PatientSex

            # 第二步：得到dicom图片所组成3D图片的维度
            ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))  # ConstPixelDims是一个元组

            # 第三步：得到x方向和y方向的Spacing并得到z方向的层厚
            ConstPixelSpacing = (
            float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

            # 第四步：得到图像的原点
            Origin = RefDs.ImagePositionPatient
            # 根据维度创建一个numpy的三维数组，并将元素类型设为：pixel_array.dtype
            ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)  # array is a numpy array

            # 第五步:遍历所有的dicom文件，读取图像数据，存放在numpy数组中
            i = 0
            for filenameDCM in lstFilesDCM:
                ds = pydicom.read_file(filenameDCM)
                image = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = image
                # cv2.imwrite("out_" + str(i) + ".png", ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)])
                i += 1

            # 第六步：对numpy数组进行转置，即把坐标轴（x,y,z）变换为（z,y,x）,这样是dicom存储文件的格式，即第一个维度为z轴便于图片堆叠
            ArrayDicom = np.transpose(ArrayDicom, (2, 0, 1))

            # 第七步：将现在的numpy数组通过SimpleITK转化为mhd和raw文件
            # ArrayDicom[ArrayDicom == 255] = 1
            sitk_img = SimpleITK.GetImageFromArray(ArrayDicom, isVector=False)

            # sitk_img.SetSpacing(ConstPixelSpacing)
            # sitk_img.SetOrigin(Origin)
            PVImagePath = os.path.join(MHDdir, MHDName)
            SimpleITK.WriteImage(sitk_img,PVImagePath)

    dataset.numOfCases = numOfCases
    dataset.save()

def DICOMtoMHDCases(PathDicom,upload_username,caseName):
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for sub in subdirList:
            if sub == 'mask':
                for maskfile in os.listdir(os.path.join(dirName, sub)):
                    patientName = dirName.split('/')[-2]
                    phase = dirName.split('/')[-1]
                    suffix = maskfile.split('.')[-1]
                    MHDdir = PathDicom + '/unzip/MHD/' + patientName
                    MHDName = phase + '_mask.' + suffix
                    if not os.path.exists(MHDdir):
                        os.makedirs(MHDdir)

                    shutil.copy(os.path.join(dirName, sub, maskfile), MHDdir)
                    txtName = PathDicom + '/unzip/DICOM/' + patientName +'/'+ patientName + '.txt'
                    print('txtName:',txtName)
                    shutil.copy(txtName, MHDdir)

        if (len(fileList) > 10 and '__MACOSX' not in dirName):
            patientName = dirName.split('/')[-3]
            phase = dirName.split('/')[-2]
            type = dirName.split('/')[-1].split('_')[0]
            MHDdir = PathDicom + '/unzip/MHD/' + patientName
            MHDName = phase + '_' + type + '.mhd'
            if not os.path.exists(MHDdir):
                os.makedirs(MHDdir)

            lstFilesDCM = []

            for filename in fileList:
                if (filename != '.DS_Store'):
                    lstFilesDCM.append(os.path.join(dirName, filename))  # 加入到列表中
            lstFilesDCM.sort()
            lstFilesDCM.reverse()


            # 第一步：将第一张图片作为参考图片，并认为所有图片具有相同维度
            RefDs = pydicom.read_file(lstFilesDCM[0])  # 读取第一张dicom图片

            patientAge = RefDs.PatientAge
            patientSex = RefDs.PatientSex

            # 第二步：得到dicom图片所组成3D图片的维度
            ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))  # ConstPixelDims是一个元组

            # 第三步：得到x方向和y方向的Spacing并得到z方向的层厚
            ConstPixelSpacing = (
            float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

            # 第四步：得到图像的原点
            Origin = RefDs.ImagePositionPatient
            # 根据维度创建一个numpy的三维数组，并将元素类型设为：pixel_array.dtype
            ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)  # array is a numpy array

            # 第五步:遍历所有的dicom文件，读取图像数据，存放在numpy数组中
            i = 0
            for filenameDCM in lstFilesDCM:
                ds = pydicom.read_file(filenameDCM)
                image = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = image
                # cv2.imwrite("out_" + str(i) + ".png", ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)])
                i += 1


            # 第六步：对numpy数组进行转置，即把坐标轴（x,y,z）变换为（z,y,x）,这样是dicom存储文件的格式，即第一个维度为z轴便于图片堆叠
            ArrayDicom = np.transpose(ArrayDicom, (2, 0, 1))

            # 第七步：将现在的numpy数组通过SimpleITK转化为mhd和raw文件
            # ArrayDicom[ArrayDicom == 255] = 1
            sitk_img = SimpleITK.GetImageFromArray(ArrayDicom, isVector=False)

            # sitk_img.SetSpacing(ConstPixelSpacing)
            # sitk_img.SetOrigin(Origin)
            PVImagePath = os.path.join(MHDdir, MHDName)
            SimpleITK.WriteImage(sitk_img,PVImagePath)

            user = models.User.objects.get(Q(username=upload_username))
            caseCreateTime = time.strftime('%Y.%m.%d/%H:%M:%S', time.localtime(time.time()))


            models.Case.objects.create(caseName=caseName,
                                       sex = patientSex,
                                       age = patientAge,
                                       caseCreateTime=caseCreateTime,
                                       PVImagePath = PVImagePath,
                                       testResult="{}",
                                       user=user)
    return PVImagePath

def ReadInfoFromDICM(PathDicom, txtPath):
    for dirName, subdirList, fileList in os.walk(PathDicom):
            lstFilesDCM = []
            for filename in fileList:
                if (filename != '.DS_Store'):
                    lstFilesDCM.append(os.path.join(dirName, filename))  # 加入到列表中
                    break
            # 第一步：将第一张图片作为参考图片，并认为所有图片具有相同维度
            RefDs = pydicom.read_file(lstFilesDCM[0])  # 读取第一张dicom图片
            patientAge = RefDs.PatientAge
            patientSex = RefDs.PatientSex

    f = open(txtPath, 'r')
    a = f.read()
    Ordinary_dict = eval(a)
    tumorType = Ordinary_dict['tumorType']

    return patientAge, patientSex, tumorType

def loadFeaturesAndLabels_Single(txtPath,featureNameList):
    f = open(txtPath, 'r')
    a = f.read()
    Ordinary_dict = eval(a)

    features = {}
    for featureName in featureNameList:
        features[featureName] = []
    for Ordinary_dictName in Ordinary_dict.keys():
        featureName = Ordinary_dictName.split('_')
        if len(featureName) >= 2:
            featureName = featureName[1]
            if featureName in featureNameList:
                if math.isnan(Ordinary_dict[Ordinary_dictName]):
                    isReturn = False
                else:
                    features[featureName].append(Ordinary_dict[Ordinary_dictName])
                    isReturn = True
        elif Ordinary_dictName == 'tumorType':
            features[Ordinary_dictName].append(Ordinary_dict[Ordinary_dictName])
        elif Ordinary_dictName == 'LBP' or Ordinary_dictName == 'BOVW':
            features[Ordinary_dictName] = Ordinary_dict[Ordinary_dictName]

    f.close()

    return isReturn, features

def loadFeaturesAndLabels_All(dirPath,featureNameList):
    dirs = os.listdir(dirPath)
    features_all = {}
    count = 0
    for featureName in featureNameList:
        features_all[featureName] = []
    for dir in dirs:
        filePath = os.path.join(dirPath,dir)
        if not filePath.endswith('.DS_Store'):
            files = os.listdir(filePath)
            for file in files:
                txtPath = os.path.join(filePath, file)
                if txtPath.endswith('.txt'):
                    _,features = loadFeaturesAndLabels_Single(txtPath,featureNameList)
                    for featureName in featureNameList:
                        features_all[featureName].append(features[featureName])
                    count += 1
                    print(count," : ",txtPath)
                    # if count >= 15:
                    #     return features_all
    return features_all


def cross_validation(clf,train_data,train_labels,cv):
    score=cross_val_score(clf,train_data,train_labels,cv=cv,n_jobs=-1,verbose=1,scoring='accuracy')
    predicted=cross_val_predict(clf,train_data,train_labels,cv=cv,n_jobs=-1,verbose=1)
    # return predicted
    return sum(score)/cv

def ML(dirPath,testNum,featureNameList,cm,tempModelPath,tempPCAPath,n_components=0.9,cv=3):
    # n_components表示PCA降维维度，cv表示交叉次数
    # featureNameList = ['shape', 'firstorder', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm', 'LBP','tumorType']
    print("selected features:", featureNameList)
    featureNameList.append('tumorType')
    features_all = loadFeaturesAndLabels_All(dirPath, featureNameList)
    featureLenDict = {}
    for name in featureNameList:
        featureLenDict[name] = len(features_all[name][0])

    min_max_scaler = preprocessing.MinMaxScaler()

    for featureName in featureNameList:
        if featureName != 'BOVW':
            if featureName == featureNameList[0]:
                basic_features = min_max_scaler.fit_transform(np.array(features_all[featureNameList[0]]))
            if featureName != 'tumorType' and featureName != featureNameList[0]:
                features_all[featureName] = min_max_scaler.fit_transform(np.array(features_all[featureName]))
                basic_features = np.hstack((np.array(basic_features), np.array(features_all[featureName])))

        else:
            allBOVWFeature = []
            allBOVWCount = []
            BOVWFeature = features_all['BOVW']
            for i in range(len(BOVWFeature)):
                allBOVWFeature.extend(BOVWFeature[i])
                allBOVWCount.append(np.shape(BOVWFeature[i]))
            print('allCount is:', allBOVWCount)
            print('allData size is ', np.shape(allBOVWFeature))
            dict = learn_dict(allBOVWFeature, 128)
            print('dict shape is:', np.shape(dict))
            representer = generate_representor(allBOVWFeature, allBOVWCount, dict)
            print('representer shape:', np.shape(representer))

    if 'BOVW' in featureNameList:
        basic_features = np.hstack((np.array(basic_features), np.array(representer)))

    # print("basic_features shape:", basic_features.shape)
    # pca = PCA(n_components=n_components)
    # # n_components=20 must be between 0 and min(n_samples, n_features)
    # basic_features = pca.fit_transform(basic_features)
    # pcaWeight = pca.explained_variance_ratio_.tolist()
    #
    # Ureduce = pca.components_
    # weight = pca.explained_variance_ratio_[np.newaxis, :]
    # orignWeight = np.matmul(weight, Ureduce)
    # featureWeight = {}
    # index = 0
    #
    # for featureName in featureNameList:
    #     if featureName != 'tumorType':
    #         featureWeight[featureName] = sum(orignWeight[0][index:index + featureLenDict[featureName]])
    #         index = index + featureLenDict[featureName]

    labels = np.array(features_all['tumorType'])
    train_features, test_features, train_labels, test_labels = train_test_split(basic_features, labels, test_size=testNum, random_state=3)

    clfs = {'svm': svm.SVC(),
            'decision_tree': tree.DecisionTreeClassifier(),
            'naive_gaussian': naive_bayes.GaussianNB(),
            'naive_mul': naive_bayes.MultinomialNB(),
            'K_neighbor': KNeighborsClassifier(),
            'bagging_knn': BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
            'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5),
            'random_forest': RandomForestClassifier(n_estimators=50),
            'adaboost': AdaBoostClassifier(n_estimators=50),
            'gradient_boost': GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1,
                                                         random_state=0)
            }

    clf = clfs[cm]
    # clf = svm.SVC()
    # clf = KNeighborsClassifier(3)
    # clf = tree.DecisionTreeClassifier(criterion='entropy')

    clf.fit(train_features, train_labels)
    predict_labels = clf.predict(train_features)
    trainAcc = metrics.accuracy_score(train_labels, predict_labels)
    print('trainAcc', trainAcc)

    predict_labels = clf.predict(test_features)
    testAcc = metrics.accuracy_score(test_labels, predict_labels)
    print("accuracy:", testAcc)

    joblib.dump(pca, tempPCAPath)
    joblib.dump(clf, tempModelPath)

    return trainAcc, testAcc, featureWeight

def testML(dirPath,testModelPath,caseName,modelName,n_components=10,cv=3): # n_components表示PCA降维维度，cv表示交叉次数
    testPCAPath = testModelPath.split('.m')[0] + 'PCA.m'
    pca = joblib.load(testPCAPath)

    featureNameList = ['shape', 'firstorder', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm']
    features_all = loadFeaturesAndLabels_All(dirPath, featureNameList)
    min_max_scaler = preprocessing.MinMaxScaler()

    for featureName in featureNameList:
        if featureName == featureNameList[0]:
            test_features = min_max_scaler.fit_transform(np.array(features_all[featureNameList[0]]))
        else:
            features_all[featureName] = min_max_scaler.fit_transform(np.array(features_all[featureName]))
            test_features = np.hstack((np.array(test_features), np.array(features_all[featureName])))

    # pca = PCA(n_components=n_components)  # n_components=20 must be between 0 and min(n_samples, n_features)

    test_features = pca.transform(test_features)

    clf = joblib.load(testModelPath)
    predict_labels = clf.predict(test_features)[0]

    findCase = models.Case.objects.get(
        Q(caseName=caseName)
    )
    pre_result = eval(findCase.testResult)
    pre_result[modelName] = predict_labels
    findCase.testResult = str(pre_result)
    findCase.save()

    # df=clf.decision_function(basic_features[0:58])

    return predict_labels

def read_mask_image(path,ww=55,wc=250, flag=False):
    if flag:
        return(MHD.MHD.rescale(MHD.MHD.read_single_file(path),ww,wc))
    return(MHD.MHD.read_single_file(path))

def transpose(image):
    [o, _, _] = np.shape(image)
    for z in range(o):
        image[z, :, :] = np.transpose(image[z, :, :])
    return image

def str2arr(str):
    str = str[1:-1]
    splits = str.split(',')
    res = []
    for split in splits:
        res.append(split[1:-1])
    return res

'''通过方差的百分比来计算将数据降到多少维是比较合适的，
函数传入的参数是特征值和百分比percentage，返回需要降到的维度数num'''
def eigValPct(eigVals,percentage):
    sortArray=sort(eigVals) #使用numpy中的sort()对特征值按照从小到大排序
    sortArray=sortArray[-1::-1] #特征值从大到小排序
    arraySum=sum(sortArray) #数据全部的方差arraySum
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i
        num+=1
        if tempSum>=arraySum*percentage:
            return num

'''pca函数有两个参数，其中dataMat是已经转换成矩阵matrix形式的数据集，列表示特征；
其中的percentage表示取前多少个特征需要达到的方差占比，默认为0.9'''
def pca(dataMat,percentage=0.9):
    meanVals=mean(dataMat,axis=0)  #对每一列求平均值，因为协方差的计算中需要减去均值
    meanRemoved=dataMat-meanVals
    covMat=cov(meanRemoved,rowvar=0)  #cov()计算方差
    eigVals,eigVects=linalg.eig(mat(covMat))  #利用numpy中寻找特征值和特征向量的模块linalg中的eig()方法
    k=eigValPct(eigVals,percentage) #要达到方差的百分比percentage，需要前k个向量
    eigValInd=argsort(eigVals)  #对特征值eigVals从小到大排序
    eigValInd=eigValInd[:-(k+1):-1] #从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    redEigVects=eigVects[:,eigValInd]   #返回排序后特征值对应的特征向量redEigVects（主成分）
    lowDDataMat=meanRemoved*redEigVects #将原始数据投影到主成分上得到新的低维数据lowDDataMat
    reconMat=(lowDDataMat*redEigVects.T)+meanVals   #得到重构数据reconMat
    return lowDDataMat,reconMat


# dirPath = "/Users/huanghuimin/PycharmProjects/App/upload/hhm/dataset5"
# trainNum = 18
# tempModelPath = "/Users/huanghuimin/PycharmProjects/App/model_save/train_TumorClassificationML.m"
# trainAcc, testAcc, _ = ML(dirPath,trainNum,tempModelPath,n_components=10,cv=3)


def readSingleFile(filePath):
    header = itk.ReadImage(filePath)
    image = itk.GetArrayFromImage(header)
    # print type(image)
    return image

def caluROI(image3D):
    indexs = np.where(image3D != 0)
    minX = np.min(indexs[:][2])
    maxX = np.max(indexs[:][2])
    minY = np.min(indexs[:][1])
    maxY = np.max(indexs[:][1])
    minZ = np.min(indexs[:][0])
    maxZ = np.max(indexs[:][0])
    # print minX, maxX
    # print minZ, maxZ
    imageROI = image3D[minZ:maxZ+1, minY:maxY+1, minX:maxX+1]
    return imageROI

def readLabels(dirPath):
    dirs = os.listdir(dirPath)
    dirs.sort(key=lambda x: int(x[-3:]))
    labels = []
    for singleDir in dirs:
        dir = os.path.join(dirPath, singleDir, 'label.txt')
        fr = open(dir, 'r+')
        dic = eval(fr.read())  # 读取的str转换为字典
        label = dic['tumorType']
        fr.close()
        labels.append(label)
    return labels

def learn_dict(allpatches, dict_size, dict_save_path=None):
        '''
        学习字典
        :param patch_path:得到patch的path
        :param dict_size: 字典的大小
        :param dict_save_path: 字典的存储路径，如果为none，则返回字典
        :return:
        '''
        # data = scio.loadmat(patch_path)
        # allpatches = []
        # for i in data.keys():
        #     if i.startswith('__'):
        #         continue
        #     patches = data[i]
        #     print np.shape(patches)
        #     indexs = range(len(patches))
        #     np.random.shuffle(indexs)
        #     allpatches.extend(patches)
        kmeans_obj = KMeans(n_clusters=dict_size, n_jobs=8, max_iter=500).fit(allpatches)
        dictionary = kmeans_obj.cluster_centers_
        dictionary = np.array(dictionary)
        if dict_save_path is not None:
            np.save(dict_save_path, dictionary)
        else:
            return dictionary

def generate_representor(all_patches, counts, dictionary):
    shape_vocabulary = np.shape(dictionary)
    vocabulary_size = shape_vocabulary[0]
    representers = []
    all_distance_arr = cal_distance(all_patches, dictionary)
    all_distance_arr = np.array(all_distance_arr)
    start = 0
    for case_index, count in enumerate(counts):
        distance_arr = all_distance_arr[start: start + len(count)]
        cur_case_representor = np.zeros([1, vocabulary_size])
        for i in range(len(distance_arr)):
            min_index = np.argmin(distance_arr[i])
            cur_case_representor[0, min_index] += 1
        representers.append(cur_case_representor.squeeze())
        start += len(count)
    return representers


def cal_distance(patches, center):
    '''
    :param patches: None 49
    :param center: 128 * 49
    :return:
    '''
    patches2 = np.multiply(patches, patches)
    center2 = np.multiply(center, center)
    patchdotcenter = np.array(np.dot(np.mat(patches), np.mat(center).T))  # None * 128
    patches2sum = np.sum(patches2, axis=1)  # None
    center2sum = np.sum(center2, axis=1)  # 128
    distance_arr = np.zeros([len(patches2sum), len(center2sum)])
    for i in range(len(patches2sum)):
        for j in range(len(center2sum)):
            distance_arr[i, j] = patches2sum[i] + center2sum[j] - 2 * patchdotcenter[i, j]
    return distance_arr