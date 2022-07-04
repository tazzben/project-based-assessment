import numpy as np
from scipy.special import expit

def probability(q, s, linear = False):
    return q + s if linear else expit(q + s)

def logisticD(q, s, question = False):
    return q * np.exp(q+s) / (1 + np.exp(q+s))**2 if question else s * np.exp(q+s) / (1 + np.exp(q+s))

def linearD(q, s, question = False):
    return q if question else s

def probabilityDerivative(q, s, question = False, linear = False):
    return linearD(q, s, question) if linear else logisticD(q, s, question)

def dItemPb(q, s, k, b, question = False, linear = False):
    return probabilityDerivative(q, s, question, linear)*(-1*(1-probability(q, s, linear))**(np.floor(k)-1))*(-1*(np.floor(k)+1)*np.floor(k/b)*(probability(q, s, linear)-1)+(np.floor(k)+1)*probability(q, s, linear)-1)

def calculateMarginal(position, data, estX, studentSize, linear = False):
    question = False if position < studentSize else True
    subData = [x for x in data if position == x[2]] if question else [x for x in data if position == x[1]]
    return np.array([ dItemPb(estX[x[1]], estX[x[2]], x[0], x[3], question, linear) for x in subData ]).mean()

def calculateMarginals(data, estX, studentSize, linear = False):
    return [ calculateMarginal(x, data, estX, studentSize, linear) for x in range(len(estX)) ]
