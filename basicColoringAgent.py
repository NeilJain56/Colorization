import imageConverter as ic
from random import sample, randint
from cv2 import cv2
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
from decimal import Decimal

def kClustering(k):
    _, cTrain = ic.getTrainingData()
    cTrain = cTrain.reshape(-1, cTrain.shape[-1])

    centers = getRandomCenters(k, cTrain)
    clusters = {}
    clusterLoss = {}
    iteration = 0
    newCenters = []
    prevLoss = 0
    currLoss = 0

    while differenceInCenters(newCenters, centers):
        if newCenters != []:
            centers = newCenters[:]

        for dataPoint in cTrain:
            center, loss = closestCenter(centers, dataPoint)
            if center not in clusters:
                clusters[center] = []
            clusters[center].append(dataPoint)

            if center not in clusterLoss:
                clusterLoss[center] = 0
            clusterLoss[center] = clusterLoss[center] + loss

        print("Iteration " + str(iteration))
        prevLoss = currLoss
        currLoss = averageLoss(centers, clusterLoss, clusters)
        if currLoss > prevLoss:
            print("INCREASE")
        print("Average Loss: " + str(currLoss))
        print(centers)
        print("")

        iteration = iteration + 1
        newCenters = recenter(clusters, centers)

        clusters.clear()
        clusterLoss.clear()

    return centers, currLoss

def recenter(clusters, centers):
    newCenters = []
    for center in centers:
        p = np.mean(clusters[center], axis=0)
        newCenters.append(tuple(p))
    return newCenters   

def closestCenter(centers, p):
    distanceToCenters = {}
    for center in centers:
        distanceToCenters[center] = dist(center, p)

    minDist = 1000
    for center in centers:
        if distanceToCenters[center] < minDist:
            minDist = distanceToCenters[center]
    
    closestCenters = []
    for center in centers:
        if distanceToCenters[center] == minDist:
            closestCenters.append(center)

    return closestCenters[randint(0, len(closestCenters) - 1)], minDist

def getRandomCenters(k, cTrain):
    centers = set(tuple(i) for i in cTrain)
    centers = list(centers)
    indexes = sample(range(0, len(centers)), k)
    centers = [centers[i] for i in indexes]
    return centers

def dist(p1, p2):
    r1 = float(p1[0])
    b1 = float(p1[1])
    g1 = float(p1[2])

    r2 = float(p2[0])
    b2 = float(p2[1])
    g2 = float(p2[2]) 

    return (((r1 - r2)**2) + ((b1 - b2)**2) + ((g1 - g2)**2))**(0.5)

def averageLoss(centers, clusterLoss, clusters):
    avgLoss = 0
    k = 0
    for center in centers:
        print("Cluster " + str(k) + " Loss: " + str(clusterLoss[center]) + " With " + str(len(clusters[center])) + " points")
        avgLoss = avgLoss + float(clusterLoss[center])
        k = k + 1
    return float(avgLoss)

def sampleTrainingData():
    numOfSamples = 100
    samples = set()

    while len(samples) != numOfSamples:
        x = randint(1, 349)
        y = randint(1, 304)
        samples.add((x, y))

    return samples

def colorImage(centers):
    gTest,cTest = ic.getTestData()
    gTrain,cTrain = ic.getTrainingData()

    for i in range(0, cTrain.shape[0]):
        for j in range(0, cTrain.shape[1]):
            cTrain[i][j] = closestCenter(centers, cTrain[i][j])[0]
    
    for i in range(0, cTest.shape[0]):
        for j in range(0, cTest.shape[1]):
            cTest[i][j] = closestCenter(centers, cTest[i][j])[0]

    for i in range(1, gTest.shape[0] - 1):
        print(i)
        for j in range(1, gTest.shape[1] - 1):
            samples = sampleTrainingData()
            similar = {}
            for s in samples:
                similar[s] = similarity(s,(i,j), gTest, gTrain)
            similar = {k: v for k, v in sorted(similar.items(), key=lambda item: item[1])}
            cTest[i][j] = getColor(similar, centers, cTrain)

    finalImage = np.concatenate((cTrain, cTest), axis=1)
    leftLoss, rightLoss = lossCalculation(cTrain, cTest)
    print("Left Side Loss from Original Image: " + str(leftLoss))
    print("Right Side Loss from Original Image: " + str(rightLoss))
    ic.displayImage(finalImage)
            
def getColor(similar, centers, cTrain):
    count = 0
    topSix = {}
    mapColorToK = {}
    for k in similar:
        if count == 6:
            break
        color = closestCenter(centers, cTrain[k[0]][k[1]])[0]

        if color not in mapColorToK:
            mapColorToK[color] = set()
        mapColorToK[color].add(k)
        
        if color not in topSix:
            topSix[color] = 0
        topSix[color] = topSix[color] + 1
        count = count + 1

    max_value = max(topSix.values())
    bestColors = []
    for color in topSix:
        if topSix[color] == max_value:
            bestColors.append(color)
    
    if len(bestColors) == 1:
        return bestColors[0]
    
    kList = []
    for color in bestColors:
        kList.extend(list(mapColorToK[color]))
    
    minDiff = None
    for k in kList:
        if minDiff == None:
            minDiff = similar[k]
        else:
            minDiff = min(minDiff, similar[k])
    
    minDiffK = []
    for k in kList:
        if similar[k] == minDiff:
            minDiffK.append(k)

    minDiffK = minDiffK[randint(0, len(minDiffK) - 1)]
    for color in mapColorToK:
        if minDiffK in mapColorToK[color]:
            return color

def similarity(train, test, gTest, gTrain):
    trainCells = getNeighbors(train)
    testCells = getNeighbors(test)

    diff = 0 
    for trainCell, testCell in zip(trainCells, testCells):
        diff = diff + ((float(gTrain[trainCell[0]][trainCell[1]][0]) - float(gTest[testCell[0]][testCell[1]][0]))**(2))**(0.5)
    return diff

def getNeighbors(p):
    x = p[0]
    y = p[1]

    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1),
            (x - 1, y - 1), (x, y)]

def roundCenters(unRoundedCenters):
    centers = []
    for c in unRoundedCenters:
        temp = []
        for val in c:
            temp.append(round(val))
        temp = tuple(temp)
        centers.append(temp)
    return centers

def lossCalculation(leftHalf, rightHalf):
    _,cTest = ic.getTestData()
    _,cTrain = ic.getTrainingData()

    cTrain = cTrain.reshape(-1, cTrain.shape[-1])
    cTest = cTest.reshape(-1, cTest.shape[-1])

    leftHalf = leftHalf.reshape(-1, leftHalf.shape[-1])
    rightHalf = rightHalf.reshape(-1, rightHalf.shape[-1])

    leftLoss = 0
    for leftPixel, trainPixel in zip(leftHalf, cTrain):
        leftLoss = leftLoss + dist(leftPixel, trainPixel)

    rightLoss = 0
    for rightPixel, testPixel in zip(rightHalf, cTest):
        rightLoss = rightLoss + dist(rightPixel, testPixel)

    return leftLoss, rightLoss

def elbowPlot(maxClusters):
    x = range(2, maxClusters + 1)
    y = []

    for k in x:
        _, loss = kClustering(k)
        y.append(loss)

    plt.plot(x, y)
    plt.show()

def differenceInCenters(newCenters, centers):
    if newCenters == []:
        return True

    for newC, c in zip(newCenters, centers):
        if dist(newC, c) > 0.5:
            return True
    return False


#centers = [(56.192328642759875, 92.4650476622787, 58.81674988651839), (16.717544793329786, 35.68589675359234, 8.699946780202236), (185.5434578241176, 191.97743782582685, 158.86753268951372), (221.13101685223182, 242.02213327054392, 242.8842207945104), (106.14859463232028, 139.3712327897976, 107.41754964786499)]
#colorImage(roundCenters(centers))
print(kClustering(6))

"""
100 Samples 5 Clusters:
Left Side Loss from Original Image: 4045378.1262451345
Right Side Loss from Original Image: 5701570.429981784

500 Samples 5 Clusters:
Left Side Loss from Original Image: 4045378.1262451345
Right Side Loss from Original Image: 5439836.102683887
"""

