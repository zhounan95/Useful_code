import os
import radiomics
import logging
import six
import numpy as np
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk
from radiomics.imageoperations import checkMask

import scipy.io as sio
from sklearn import preprocessing

from sklearn import svm
from sklearn.linear_model import RidgeClassifier, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn import metrics
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
from Tools.tools import *
np.set_printoptions(threshold=np.inf)

def caluLBP(images, eps=1e-7):
    lbps = []
    [z, y, x] = np.shape(images)
    for i in range(z):
        curImage = images[i, :, :]
        r = 3
        points = 8 * 3
        lbp = local_binary_pattern(curImage, points, r, 'uniform')
        lbps.extend(lbp)
    n_bins = 26
    (hists, _) = np.histogram(lbps, n_bins)
    histogram = hists.astype("float")
    histogram /= (histogram.sum() + eps)
    return histogram

def get_patch(image, patch_size, patch_step, data_dir=None, singledir=None):
    [z, y, x] = np.shape(image)
    r = patch_size // 2
    patch_count = 0
    case_patch = []
    for k in range(z):
        for j in range(r, y - r, patch_step):
            for i in range(r, x - r, patch_step):
                cur_patch = image[k][j - r: j + r + 1, i - r: i + r + 1]
                if(np.sum(cur_patch) == 0):
                    continue
                if(data_dir != None):
                    save_path = os.path.join(data_dir,  singledir + '_' + str(patch_count) + '.npy')
                    np.save(save_path, np.array(cur_patch))
                case_patch.append(cur_patch.flatten())
                patch_count += 1
    print('case_patch is:', np.shape(case_patch))
    return np.array(case_patch), patch_count

def findTumorType(labelPath):
    f = open(labelPath, 'r')
    a = f.read()
    Dict = eval(a)
    tumorType = Dict['tumorType']
    return tumorType

def featureExtract_all(PathDicom,train=True):
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for sub in subdirList:
            if sub == 'MHD':
                dirPath = os.path.join(dirName,sub)
                patientNames = os.listdir(dirPath)
                for patientName in patientNames:
                    if not patientName.endswith('.DS_Store'):
                        patientPath = os.path.join(dirPath,patientName)
                        mhds = os.listdir(patientPath)
                        for mhd in mhds:
                            if "image" in mhd and mhd.endswith('.mhd'):
                                imageName = os.path.join(patientPath,mhd)
                            elif "Mask" in mhd and mhd.endswith('.mhd'):
                                maskName = os.path.join(patientPath,mhd)
                        if train == True:
                            labelPath = patientPath.replace("MHD", "DICOM") + '/label.txt'
                            print(labelPath)
                            tumorType = findTumorType(labelPath)
                            featureExtract(imageName, maskName,tumorType)
                        else:
                            featureExtract(imageName, maskName)

def featureExtract(imageName,maskName,tumorType=None):
    '''
        # 在helloRadiomics中要自定义setting，而此处我用了exampleSettings中的Params.yaml
        # Define settings for signature calculation
        # These are currently set equal to the respective default values
        settings = {}
        settings['binWidth'] = 25
        settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
        settings['interpolator'] = sitk.sitkBSpline

        # Initialize feature extractor
        extractor = featureextractor.RadiomicsFeaturesExtractor(**settings)
    '''
    patientID = imageName.split('/')[-2]
    dirName = imageName.split(patientID)[0]

    txtName = patientID + '_.txt'
    txtPath = os.path.join(dirName, patientID, txtName)

    f = open(txtPath, 'w')

    # Get the location of the example settings file
    paramsFile = os.path.abspath(os.path.join('exampleSettings', 'Params.yaml'))

    if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
        print('Error getting testcase!')
        exit()

    # Initialize feature extractor using the settings file
    extractor = featureextractor.RadiomicsFeaturesExtractor(paramsFile)

    '''
        # 在helloRadiomics中，通过enableFeaturesByName定义我只需要计算的类
        # Only enable mean and skewness in firstorder
        extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])
        # 而在此处，我要计算全部的特征
    '''

    print("Calculating features")
    featureVector = extractor.execute(imageName, maskName)
    Ordinary_dict = {}
    for featureName in featureVector.keys():
        Ordinary_dict[featureName] = featureVector[featureName]

    pvLiverImages = readSingleFile(imageName)
    pvTumorMask = readSingleFile(maskName)
    pvImagesROI = pvLiverImages * pvTumorMask
    LBPFeature = caluLBP(pvImagesROI)
    Ordinary_dict['LBP'] = LBPFeature

    # BOVWFeature, count = get_patch(pvImagesROI, 7, 1)
    # Ordinary_dict['BOVW'] = BOVWFeature
    # print('BOVW feature shape:', Ordinary_dict['BOVW'].shape)


    if tumorType!= None:
        Ordinary_dict['tumorType'] = tumorType
    print(Ordinary_dict)

    f.write(str(Ordinary_dict))
    f.close()

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
                features[featureName].append(Ordinary_dict[Ordinary_dictName])
        elif Ordinary_dictName == 'tumorType':
            features[Ordinary_dictName].append(Ordinary_dict[Ordinary_dictName])

    f.close()

    return features

def loadFeaturesAndLabels_All(dirPath,featureNameList):
    dirs = os.listdir(dirPath)
    features_all = {}
    for featureName in featureNameList:
        features_all[featureName] = []
    for dir in dirs:
        filePath = os.path.join(dirPath,dir)
        if not filePath.endswith('.DS_Store'):
            files = os.listdir(filePath)
            for file in files:
                txtPath = os.path.join(filePath, file)
                if txtPath.endswith('.txt'):
                    features= loadFeaturesAndLabels_Single(txtPath,featureNameList)
                    for featureName in featureNameList:
                        features_all[featureName].append(features[featureName])

    return features_all


def cross_validation(clf,train_data,train_labels,cv):
    score=cross_val_score(clf,train_data,train_labels,cv=cv,n_jobs=-1,verbose=1,scoring='accuracy')
    predicted=cross_val_predict(clf,train_data,train_labels,cv=cv,n_jobs=-1,verbose=1)
    print(score)
    print('cross validation score:',sum(score)/cv)
    print(predicted)
    # return predicted
    return sum(score)/cv

def ML(dirPath,n_components=2,cv=2): # n_components表示PCA降维维度，cv表示交叉次数
    featureNameList = ['shape', 'firstorder', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm', 'LBP','tumorType']
    features_all = loadFeaturesAndLabels_All(dirPath, featureNameList)
    min_max_scaler = preprocessing.MinMaxScaler()

    for featureName in featureNameList:
        if featureName == featureNameList[0]:
            basic_features = min_max_scaler.fit_transform(np.array(features_all[featureNameList[0]]))
        if featureName != 'tumorType' and featureName != featureNameList[0]:
            features_all[featureName] = min_max_scaler.fit_transform(np.array(features_all[featureName]))
            basic_features = np.hstack((np.array(basic_features), np.array(features_all[featureName])))

    pca = PCA(n_components=n_components)  # n_components=20 must be between 0 and min(n_samples, n_features)
    print(pca)
    print("basic_features shape:", basic_features.shape)
    basic_features = pca.fit_transform(basic_features)
    print("basic_features shape:", basic_features.shape)
    print(pca.explained_variance_ratio_)

    train_labels = np.array(features_all['tumorType'])

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    cross_validation(clf, basic_features, train_labels, cv=cv)

if __name__ == '__main__':

    # featureExtract_all('/data/medical_system/App_hhm/App/upload/hhm/dataTest')

    featureExtract_all('/data/medical_system/App_hhm/App/upload/hhm/datasetAll/unzip/', train=True)

    # ML(dirPath)


