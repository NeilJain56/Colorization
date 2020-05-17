from Node import Node
import numpy as np
from copy import deepcopy
from random import shuffle

class NeuralNet:
    def __init__(self):
        self.layers = []
        self.createNetwork()
        self.gradients = []

    def createNetwork(self):
        biasNode = Node(0)
        biasNode.setOutput(1)
        self.layers.append([biasNode])
        self.layers[0].extend([Node(0) for i in range(9)])

        biasNode = Node(0)
        biasNode.setOutput(1)
        self.layers.append([biasNode])
        self.layers[1].extend([Node(10) for i in range(5)])

        self.layers.append([])
        self.layers[2].extend([Node(6) for i in range(3)])

    def getOutputOfLayer(self, layerNum):
        output = []
        for n in self.layers[layerNum]:
            output.append(n.out)
        return np.array(output)
    
    def getNetworkOutput(self, x):
        self.forwardPropagation(x)
        return self.getOutputOfLayer(len(self.layers) - 1)

    def forwardPropagation(self, x):
        for i in range(1, len(self.layers[0])):
            self.layers[0][i].setOutput(x[i - 1])

        for i in range(1, len(self.layers) - 1):
            for j in range(1, len(self.layers[i])):
                self.layers[i][j].calculateOutput(self.getOutputOfLayer(i-1))
        
        lastLayer = len(self.layers) - 1
        for i in range(0, len(self.layers[lastLayer])):
            self.layers[lastLayer][i].calculateOutput(self.getOutputOfLayer(lastLayer-1))

    def backPropagation(self, y):

        colorWeights = [2, 1, 3]
        lastLayer = len(self.layers) - 1
        for i in range(0, len(self.layers[lastLayer])):
            out = self.layers[lastLayer][i].out
            self.layers[lastLayer][i].lossNodeDerivative = 2*(out - y[i])*colorWeights[i]
            self.layers[lastLayer][i].calculateWeightDerivatives(self.getOutputOfLayer(lastLayer - 1))

        for i in range(lastLayer - 1, 0, -1):
            for j in range(1, len(self.layers[i])):
                self.layers[i][j].calculateNodeDerivative(self.layers[i+1], self.getOutputOfLayer(i), j)
                self.layers[i][j].calculateWeightDerivatives(self.getOutputOfLayer(i - 1))

        weightDerivatives = {}
        weightDerivatives[lastLayer] = []
        for i in range(0, len(self.layers[lastLayer])):
            weightDerivatives[lastLayer].append(self.layers[lastLayer][i].lossWeightDerivatives)

        for i in range(lastLayer - 1, 0, -1):
            weightDerivatives[i] = []
            for j in range(1, len(self.layers[i])):
                weightDerivatives[i].append(self.layers[i][j].lossWeightDerivatives)
        
        return deepcopy(weightDerivatives)

    def updateDerivatives(self, weightDerivatives):
        lastLayer = len(self.layers) - 1
        for i in range(0, len(self.layers[lastLayer])):
            self.layers[lastLayer][i].lossWeightDerivatives = weightDerivatives[lastLayer][i]
            self.layers[lastLayer][i].updateWeights()

        del weightDerivatives[lastLayer]

        for i in range(1, lastLayer):
            for j in range(1, len(self.layers[i])):
                self.layers[i][j].lossWeightDerivatives = weightDerivatives[i][j-1]
                self.layers[i][j].updateWeights()

    def miniBatch(self, gTrain, cTrain):
        gradients = []
        for x, y in zip(gTrain, cTrain):
            self.forwardPropagation(x)
            gradient = self.backPropagation(y)
            gradients.append(gradient)
        return self.averageGradients(gradients)

    def averageGradients(self, gradients):
        g1 = gradients[0]
        for l in g1:
            for i in range(len(g1[l])):
                temp = []
                for gradient in gradients:
                    temp.append(gradient[l][i])
                g1[l][i] = np.mean(temp, axis=0)
        return deepcopy(g1)

    def train(self, processedGrayTrainData, processedColorTrainData, processedGrayLossData, processedColorLossData):
        batchSize = 50
        lowBatch = 0
        highBatch = batchSize
        prevLoss = None
        currLoss = None
        num = 0
        validationLoss = []
        vL = self.calculateLoss(processedGrayLossData, processedColorLossData)
        validationLoss.append(vL)
        trainingLoss = []
        tL = self.calculateLoss(processedGrayTrainData, processedColorTrainData)
        trainingLoss.append(tL)
        epochs = 0
        processedGrayTrainData, processedColorTrainData = self.shuffleData(processedGrayTrainData, processedColorTrainData)

        while (prevLoss == None or currLoss < prevLoss) and epochs < 40:
            avgGradient = self.miniBatch(processedGrayTrainData[lowBatch:highBatch, :], processedColorTrainData[lowBatch:highBatch, :])
            self.updateDerivatives(avgGradient)
            num = num + 1
            if highBatch == processedGrayTrainData.shape[0]:
                epochs = epochs + 1
                prevLoss = currLoss
                currLoss = self.calculateLoss(processedGrayLossData, processedColorLossData)
                validationLoss.append(currLoss)
                trainingSetLoss = self.calculateLoss(processedGrayTrainData, processedColorTrainData)
                trainingLoss.append(trainingSetLoss)

                print("Epoch " + str(epochs))
                print("Validation Set Loss: " + str(validationLoss))
                print("Training Set Loss: " + str(trainingLoss))
                print("")

                processedGrayTrainData, processedColorTrainData = self.shuffleData(processedGrayTrainData, processedColorTrainData)
                lowBatch = 0
                highBatch = batchSize
            else:
                lowBatch = highBatch + 1
                highBatch = highBatch + batchSize

        print(validationLoss)
        print(trainingLoss)

    def calculateLoss(self, processedGrayLossData, processedColorLossData):
        loss = 0
        for gData, cData in zip(processedGrayLossData, processedColorLossData):
            self.forwardPropagation(gData)
            diff = np.subtract(self.getOutputOfLayer(len(self.layers) - 1), cData)
            sqr = np.square(diff)
            loss = loss + np.sum(sqr)
        return loss/len(processedGrayLossData)

    def shuffleData(self, processedGrayTrainData, processedColorTrainData):
        temp = list(zip(processedGrayTrainData, processedColorTrainData))
        shuffle(temp)
        processedGrayTrainData, processedColorTrainData = zip(*temp)
        processedGrayTrainData = np.array(processedGrayTrainData)
        processedColorTrainData = np.array(processedColorTrainData)
        return processedGrayTrainData, processedColorTrainData


    

        



        


        
                

        
        


        

    