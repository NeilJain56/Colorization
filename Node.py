import numpy as np
import math
from random import gauss

class Node:
    def __init__(self, numOfWeights):
        self.weights = []
        self.initializeWeights(numOfWeights)
        self.learningRate = 0.01
        self.out = 0
        self.lossNodeDerivative = 0
        self.lossWeightDerivatives = []

    def initializeWeights(self, numOfWeights):
        if numOfWeights != 0:
            mean = 0
            variance = 1.0/numOfWeights
            self.weights = [gauss(mean, math.sqrt(variance)) for i in range(numOfWeights)]

    def setOutput(self, inp):
        self.out = inp
    
    def calculateOutput(self, inp):
        self.out = self.sigma(np.dot(self.weights, inp))

    def sigma(self, val):
        bottom = 1 + math.exp(-1*float(val))
        return 1.0 / float(bottom)

    def derivativeOfSigma(self, val):
        return self.sigma(val)*(1 - self.sigma(val))

    def calculateNodeDerivative(self, nextLayer, output, k):
        total = 0
        for node in nextLayer:
            d = self.derivativeOfSigma(np.dot(node.weights, output))
            total = total + node.lossNodeDerivative*d*node.weights[k]
        self.lossNodeDerivative = total

    def calculateWeightDerivatives(self, outputPrev):
        self.lossWeightDerivatives = []
        d = self.derivativeOfSigma(np.dot(self.weights, outputPrev))
        d = d*self.lossNodeDerivative
        for i in range(len(self.weights)):
            self.lossWeightDerivatives.append(d*outputPrev[i])

    def updateWeights(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - (self.learningRate*self.lossWeightDerivatives[i])
    
    