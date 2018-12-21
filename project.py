import glob, os
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
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

X, Y = dataset(imlist)
plt.imshow(X[0])
plt.title(Y[0])
plt.show()

