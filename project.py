import glob, os
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 

### Reading Data ###
path = "images/"
imlist = glob.glob(os.path.join(path, "*.jpg"))

def dataset(file_list, size=(180, 300), flattened = False):
    data = []
    for i, file in enumerate(file_list):
        image = cv.imread(file)
        image2 = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image = cv.resize(image2, size)
        if flattened:
            image = image.flatten()
        data.append(image)

    labels = [1 if f.split("/")[-1][0] == 'P' else 0 for f in file_list]
    return np.array(data), np.array(labels)

print('Reading data...')
x, y = dataset(imlist, flattened=True)

### Split data ###
print('Splitting data...')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

'''### Train SVCs and test ###
print('Training LinearSVC...')
clf = svm.LinearSVC()
clf.fit(x_train, y_train)
print('LinearSVC Score: ', clf.score(x_test, y_test))

print('Training SVC...')
clf = svm.SVC(gamma='scale')
clf.fit(x_train, y_train)
print('SVC Score: ', clf.score(x_test, y_test))'''

### Apply PCA ###
pca = PCA()
pca.fit(x_train)
print(len(pca.explained_variance_ratio_))
index = 0
sum_ = 0
for i in range(len(pca.explained_variance_ratio_)):
    if sum_ > 0.90:
        index = i
        break
    sum_+=pca.explained_variance_ratio_[i]
print("90 percent explained variance coverage component index: ", index) 

pca = PCA(index)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)

'''### Train SVCs and test using transformed data ###
print('Training LinearSVC...')
clf = svm.LinearSVC()
clf.fit(x_train, y_train)
print('LinearSVC Score: ', clf.score(x_test, y_test))

print('Training SVC...')
clf = svm.SVC(gamma='scale')
clf.fit(x_train, y_train)
print('SVC Score: ', clf.score(x_test, y_test))'''






