from Node import Node
from NeuralNet import NeuralNet
import imageConverter as ic
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from random import shuffle

def preprocessTrainingData(gTrain, cTrain):
    processedGrayTrainData = []
    processedColorTrainData = []
    for i in range(1, cTrain.shape[0] - 1):
        for j in range(1, cTrain.shape[1] - 1):
            processedGrayTrainData.append([val/255.0 for val in getGrayScaleNeighbors((i, j), gTrain)])
            processedColorTrainData.append([val/255.0 for val in cTrain[i][j]])
    
    processedGrayTrainData = np.array(processedGrayTrainData)
    processedColorTrainData = np.array(processedColorTrainData)

    processedGrayTrainData, processedColorTrainData = shuffleData(processedGrayTrainData, processedColorTrainData)
    
    processedGrayLossData = processedGrayTrainData[84001:106096, :]
    processedColorLossData = processedColorTrainData[84001:106096, :]

    processedGrayTrainData = processedGrayTrainData[0:84000, :]
    processedColorTrainData = processedColorTrainData[0:84000, :]

    return processedGrayTrainData, processedColorTrainData, processedGrayLossData, processedColorLossData

def shuffleData(processedGrayTrainData, processedColorTrainData):
        temp = list(zip(processedGrayTrainData, processedColorTrainData))
        shuffle(temp)
        processedGrayTrainData, processedColorTrainData = zip(*temp)
        processedGrayTrainData = np.array(processedGrayTrainData)
        processedColorTrainData = np.array(processedColorTrainData)
        return processedGrayTrainData, processedColorTrainData

def getGrayScaleNeighbors(p, gTrain):
    x = p[0]
    y = p[1]
    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1),
            (x - 1, y - 1), (x, y)]

    return [gTrain[x][y][0] for x,y, in neighbors]

def trainNetwork():
    gTrain, cTrain = ic.getTrainingData()
    processedGrayTrainData, processedColorTrainData, processedGrayLossData, processedColorLossData = preprocessTrainingData(gTrain, cTrain)
    neuralNetwork = NeuralNet()
    neuralNetwork.train(processedGrayTrainData, processedColorTrainData, processedGrayLossData, processedColorLossData)
    return neuralNetwork

def processEvalTestData(gTest, neuralNetwork):
    gTest = gTest.tolist()
    finalResult = deepcopy(gTest)
    for i in range(1, len(gTest) - 1):
        for j in range(1, len(gTest[i]) - 1):
            temp = neuralNetwork.getNetworkOutput([val/255.0 for val in getGrayScaleNeighbors((i,j), gTest)])
            finalResult[i][j] = [int(round(val*255.0)) for val in temp]
    return finalResult

def colorImage():
    neuralNetwork = trainNetwork()
    gTest, cTest = ic.getTestData()
    _, cTrain = ic.getTrainingData()
    gTest = processEvalTestData(gTest, neuralNetwork)
    gTest = np.array(gTest)
    print("Loss for Right: " + str(lossForRightHalf(gTest, cTest)))
    finalImage = np.concatenate((cTrain, gTest), axis=1)
    ic.displayImage(finalImage)

def lossForRightHalf(gTest, cTest):
    totalLoss = 0
    for gR, cR in zip(gTest, cTest):
        for gC, cC in zip(gR, cR):
            totalLoss = totalLoss + dist(gC, cC)
    return totalLoss

def dist(p1, p2):
    r1 = float(p1[0])
    b1 = float(p1[1])
    g1 = float(p1[2])

    r2 = float(p2[0])
    b2 = float(p2[1])
    g2 = float(p2[2]) 

    return (((r1 - r2)**2) + ((b1 - b2)**2) + ((g1 - g2)**2))**(0.5)

def testForwardProp():
    neuralNet = NeuralNet()
    neuralNet.forwardPropagation([1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1])
    print(neuralNet.getOutputOfLayer(1))

def testBackwardProp():
    neuralNet = NeuralNet()
    neuralNet.forwardPropagation([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    print("Layer 2: " + str(neuralNet.getOutputOfLayer(2)))
    print("Layer 1: " + str(neuralNet.getOutputOfLayer(1)))
    print(neuralNet.backPropagation([0.5, 0.5, 0.5]))

def testMiniBatch():
    neuralNet = NeuralNet()
    gTrain = [[1, 1, 1, 1, 1, 1, 1, 1, 1], 
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [1, 1, 1, 0.5, 0.5, 0.5, 1, 1, 1]]
    cTrain = [[0.5, 0.5, 0.5], [0.4, 0.4, 0.4], [0.3, 0.3, 0.3]]

    for x, y in zip(gTrain, cTrain):
        neuralNet.forwardPropagation(x)
        print(neuralNet.backPropagation(y))
        print("------------------------------")
    
    print(neuralNet.miniBatch(gTrain, cTrain))

def testUpdateWeights():
    neuralNet = NeuralNet()
    gTrain = [[1, 1, 1, 1, 1, 1, 1, 1, 1], 
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [1, 1, 1, 0.5, 0.5, 0.5, 1, 1, 1]]
    cTrain = [[0.5, 0.5, 0.5], [0.4, 0.4, 0.4], [0.3, 0.3, 0.3]]
    w = neuralNet.miniBatch(gTrain, cTrain)
    print(w)
    print("------------------------------")
    neuralNet.updateDerivatives(w)
    print(neuralNet.layers[1][1].weights)

def plotMSE():
    validationLoss = [0.37810995773500583, 0.2998517939658667, 0.1578829853897568, 0.08931938070900826, 0.06370167439715697, 0.052582280036626954, 0.04671209484385662, 0.04300707912268373, 0.04036052687093037, 0.038306451967107774, 0.03662837555953694, 0.03522369973237779, 0.034030824308898366, 0.03299958616716287, 0.03210581760630804, 0.031331111035898716, 0.030650815809378125, 0.030050810321245985, 0.029540194819025308, 0.029066182668727854, 0.028659498389067197, 0.028297327178577452, 0.02797960952114814, 0.027693638021977964, 0.02744416344091988, 0.027220448204796292, 0.027017752099396025, 0.026840744352653965, 0.026682915253876158, 0.026535326711257485, 0.02641193444694165, 0.026289673302201784, 0.02618297746970596, 0.026089622411125883, 0.02600219914537798, 0.025921197629425442, 0.025851838991969682, 0.025784566795376357, 0.025727315441872272, 0.02566927753662265]
    trainingLoss = [0.37647491224872365, 0.29836834644695226, 0.15680652478637658, 0.08842359133151842, 0.06294235616277785, 0.05191593569090454, 0.04614588967540463, 0.04250335990672409, 0.03993492811139976, 0.03793844673038555, 0.036302401462517356, 0.03494208619788473, 0.03378501942575322, 0.032776447189130725, 0.031916298567850956, 0.031166854082467874, 0.0305064863292333, 0.02991597715749675, 0.029426723873826725, 0.0289598962260427, 0.02857018281794861, 0.028215607563091535, 0.027906612711157117, 0.02763000436636362, 0.027384123875055302, 0.02717047322345105, 0.026975829394924312, 0.02680250542583914, 0.02664488304565095, 0.026505117517634846, 0.026382243441182004, 0.026266131499842606, 0.026164244923480066, 0.0260716220463215, 0.02598918371650184, 0.02591332308895923, 0.025842764158271074, 0.02578220668520487, 0.025722404868864023, 0.025667476397996866]
    validationLoss = np.array(validationLoss)
    trainingLoss = np.array(trainingLoss)

    x = range(0, 40)

    plt.plot(x, validationLoss, label="Validation MSE")
    plt.plot(x, trainingLoss, label="Training MSE")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("Number of Epochs vs. Mean Squared Error of Validation and Training Set")
    plt.legend()
    plt.show()

def plotRatio():
    validationLoss = [0.37810995773500583, 0.2998517939658667, 0.1578829853897568, 0.08931938070900826, 0.06370167439715697, 0.052582280036626954, 0.04671209484385662, 0.04300707912268373, 0.04036052687093037, 0.038306451967107774, 0.03662837555953694, 0.03522369973237779, 0.034030824308898366, 0.03299958616716287, 0.03210581760630804, 0.031331111035898716, 0.030650815809378125, 0.030050810321245985, 0.029540194819025308, 0.029066182668727854, 0.028659498389067197, 0.028297327178577452, 0.02797960952114814, 0.027693638021977964, 0.02744416344091988, 0.027220448204796292, 0.027017752099396025, 0.026840744352653965, 0.026682915253876158, 0.026535326711257485, 0.02641193444694165, 0.026289673302201784, 0.02618297746970596, 0.026089622411125883, 0.02600219914537798, 0.025921197629425442, 0.025851838991969682, 0.025784566795376357, 0.025727315441872272, 0.02566927753662265]
    trainingLoss = [0.37647491224872365, 0.29836834644695226, 0.15680652478637658, 0.08842359133151842, 0.06294235616277785, 0.05191593569090454, 0.04614588967540463, 0.04250335990672409, 0.03993492811139976, 0.03793844673038555, 0.036302401462517356, 0.03494208619788473, 0.03378501942575322, 0.032776447189130725, 0.031916298567850956, 0.031166854082467874, 0.0305064863292333, 0.02991597715749675, 0.029426723873826725, 0.0289598962260427, 0.02857018281794861, 0.028215607563091535, 0.027906612711157117, 0.02763000436636362, 0.027384123875055302, 0.02717047322345105, 0.026975829394924312, 0.02680250542583914, 0.02664488304565095, 0.026505117517634846, 0.026382243441182004, 0.026266131499842606, 0.026164244923480066, 0.0260716220463215, 0.02598918371650184, 0.02591332308895923, 0.025842764158271074, 0.02578220668520487, 0.025722404868864023, 0.025667476397996866]
    validationLoss = np.array(validationLoss)
    trainingLoss = np.array(trainingLoss)

    x = range(0, 40)

    ratio = np.divide(validationLoss, trainingLoss)

    plt.plot(x, ratio)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Ratio of Validation to Training MSE")
    plt.title("Number of Epochs vs. Ratio of Validation to Training MSE")
    plt.show()

colorImage()

"""
Loss for Right: 3978235.846565175
"""


