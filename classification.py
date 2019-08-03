# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
np.set_printoptions(threshold=np.inf)

class KNN:
    @staticmethod
    def kCrossValidation(allData, Label, count):
        kf = KFold(n_splits=5, shuffle=True)
        accuracy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for trainIndex, testIndex in kf.split(Label):
            trainData, trainLabel, trainCount, testData, testLabel, testCount =KNN.getTrainTestData(allData, Label, count,
                                                                                                 trainIndex, testIndex)
            a = KNN.myKNN(trainData, trainLabel, testData, testLabel)
            accuracy += a
        accuracy = np.array(accuracy) / 5
        print ('average Accuracy is ', accuracy)
        print ('allAverageTrueAccuracy is ', np.sum(accuracy)/9)
        plt.plot(accuracy)
        plt.show()

    @staticmethod
    def getTrainTestData(allData, Label, count, trainIndexs, testIndexs):
        trainData = []
        trainLabel = []
        trainCount = []
        testData = []
        testCount = []
        testLabel = []
        for i in range(count):
            if i in trainIndexs:
                trainLabel.append(Label[i])
                trainCount.append(i)
                trainData.append(allData[i])
            elif i in testIndexs:
                testLabel.append(Label[i])
                testCount.append(i)
                testData.append(allData[i])
            else:
                print('error')
        return trainData, trainLabel, trainCount, testData, testLabel, testCount

    @staticmethod
    def myKNN(trainData, trainLabel, testData, testLabel):
        accuracy = []
        for i in range(9):
            model = KNeighborsClassifier(n_neighbors=(i + 1))
            model.fit(trainData, trainLabel)
            predictRes = model.predict(testData)
            accuracy.append(metrics.accuracy_score(testLabel, predictRes))  # 每个case结果准确率
            print ('accuracy is ', accuracy)
        return accuracy

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR

class SVM:
    @staticmethod
    def mySVR(trainData, trainLabel, testData, testLabel):
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        model = OneVsRestClassifier(svr_rbf, -1)
        model.fit(trainData, trainLabel)
        predictRes = model.predict(testData)
        accuracy = metrics.accuracy_score(testLabel, predictRes)
        return accuracy

    @staticmethod
    def kCrossValidation(allData, Label, count):
        kf = KFold(n_splits=5, shuffle=True)
        accuracy = []
        for trainIndex, testIndex in kf.split(Label):
            trainData, trainLabel, trainCount, testData, testLabel, testCount = SVM.getTrainTestData(allData, Label,
                                                                                                     count, trainIndex,
                                                                                                     testIndex)
            a = SVM.mySVR(trainData, trainLabel, testData, testLabel)
            accuracy.append(a)
        print (accuracy)
        aveAccuracy = np.sum(accuracy) / 5
        print ('average Accuracy is ', aveAccuracy)
        plt.plot(accuracy)
        plt.show()

    @staticmethod
    def getTrainTestData(allData, Label, count, trainIndexs, testIndexs):
        trainData = []
        trainLabel = []
        trainCount = []
        testData = []
        testCount = []
        testLabel = []
        for i in range(count):
            if i in trainIndexs:
                trainLabel.append(Label[i])
                trainCount.append(i)
                trainData.append(allData[i])
            elif i in testIndexs:
                testLabel.append(Label[i])
                testCount.append(i)
                testData.append(allData[i])
            else:
                print ('error')
        return trainData, trainLabel, trainCount, testData, testLabel, testCount

from sklearn.ensemble import RandomForestClassifier

class RF:

    @staticmethod
    def kCrossValidation(allData, Label, count):
        kf = KFold(n_splits=5, shuffle=True)
        accuracy = []
        for trainIndex, testIndex in kf.split(Label):
            trainData, trainLabel, trainCount, testData, testLabel, testCount = RF.getTrainTestData(allData, Label,
                                                                                                     count,
                                                                                                     trainIndex,
                                                                                                     testIndex)
            a = SVM.mySVR(trainData, trainLabel, testData, testLabel)
            accuracy.append(a)
        print (accuracy)
        aveAccuracy = np.sum(accuracy) / 5
        print ('average Accuracy is ', aveAccuracy)
        plt.plot(accuracy)
        plt.show()

    @staticmethod
    def getTrainTestData(allData, Label, count, trainIndexs, testIndexs):
        trainData = []
        trainLabel = []
        trainCount = []
        testData = []
        testCount = []
        testLabel = []
        for i in range(count):
            if i in trainIndexs:
                trainLabel.append(Label[i])
                trainCount.append(i)
                trainData.append(allData[i])
            elif i in testIndexs:
                testLabel.append(Label[i])
                testCount.append(i)
                testData.append(allData[i])
            else:
                print ('error')
        return trainData, trainLabel, trainCount, testData, testLabel, testCount

    @staticmethod
    def myRF(trainData, trainLabel, testData, testLabel):
        # Create the model with 100 trees
        model = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
        # Fit on training data
        model.fit(trainData, trainLabel)
        # Actual class predictions
        predictRes = model.predict(testData)
        # Probabilities for each class
        # rf_probs = model.predict_proba(testLabel)[:, 1]
        accuracy = metrics.accuracy_score(testLabel, predictRes)
        return accuracy

# from feature_extract import LBP

'''LBP+KNN'''
# if __name__ == '__main__':
#     dirPath = '/home/zhounan/Documents/ML_Feature/dataset/datasetAll'
#     allData, allCount, label =LBP.extractFeature(dirPath)
#     print np.sum(allData[0])
#     KNN.kCrossValidation(allData, label, allCount)

'''LBP+SVM'''
# if __name__ == '__main__':
#     dirPath = '/home/zhounan/Documents/ML_Feature/dataset/datasetAll'
#     allData, allCount, label =LBP.extractFeature(dirPath)
#     SVM.kCrossValidation(allData, label, allCount)

'''LBP+RF'''
# if __name__ == '__main__':
#     dirPath = '/home/zhounan/Documents/ML_Feature/dataset/datasetAll'
#     allData, allCount, label =LBP.extractFeature(dirPath)
#     RF.kCrossValidation(allData, label, allCount)

from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier


x_train, y_train, x_test, y_test = load_data()

clfs = {'svm': svm.SVC(),\
        'decision_tree':tree.DecisionTreeClassifier(),
        'naive_gaussian': naive_bayes.GaussianNB(), \
        'naive_mul':naive_bayes.MultinomialNB(),\
        'K_neighbor' : neighbors.KNeighborsClassifier(),\
        'bagging_knn' : BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5,max_features=0.5), \
        'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5,max_features=0.5),
        'random_forest' : RandomForestClassifier(n_estimators=50),\
        'adaboost':AdaBoostClassifier(n_estimators=50),\
        'gradient_boost' : GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0)
        }

def try_different_method(clf):
    clf.fit(x_train,y_train.ravel())
    score = clf.score(x_test,y_test.ravel())
    print('the score is :', score)

for clf_key in clfs.keys():
    print('the classifier is :',clf_key)
    clf = clfs[clf_key]
    try_different_method(clf)