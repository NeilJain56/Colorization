import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv2

def showColorImage():
    im = cv2.imread("img/tropical.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im, interpolation="none")
    plt.show()

def showGrayImage():
    im = cv2.imread("img/grayTropical.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im, interpolation="nearest")
    plt.show()

def getTrainingData():
    colorData = cv2.imread("img/tropical.png")
    colorData = cv2.cvtColor(colorData, cv2.COLOR_BGR2RGB)
    grayData = cv2.imread("img/grayTropical.png")
    grayData = cv2.cvtColor(grayData, cv2.IMREAD_GRAYSCALE)
    grayData = grayData[:, :, :-1]

    rightBound = colorData.shape[1]//2

    colorLeftData = colorData[:,0:rightBound,:]
    grayLeftData = grayData[:,0:rightBound,:]
    return grayLeftData, colorLeftData

def getTestData():
    colorData = cv2.imread("img/tropical.png")
    colorData = cv2.cvtColor(colorData, cv2.COLOR_BGR2RGB)
    grayData = cv2.imread("img/grayTropical.png")
    grayData = cv2.cvtColor(grayData, cv2.IMREAD_GRAYSCALE)
    grayData = grayData[:, :, :-1]

    leftBound = colorData.shape[1]//2
    rightBound = colorData.shape[1]

    colorRightData = colorData[:,leftBound:rightBound,:]
    grayRightData = grayData[:,leftBound:rightBound,:]
    return grayRightData, colorRightData

def displayImage(arr):
    plt.imshow(arr)
    plt.show()
    
