import numpy as np
import pandas as pd
from scipy.special import expit

def probability(q, s, linear = False):
    return q + s if linear else expit(q + s)

def logisticD(q, s, question = False):    
    return q * expit(q + s)*(1-expit(q + s)) if question else s * expit(q + s)*(1 - expit(q + s))

def linearD(q, s, question = False):
    return q if question else s

def probabilityDerivative(q, s, question = False, linear = False):
    return linearD(q, s, question) if linear else logisticD(q, s, question)

def itemPb(q, s, k, b, linear = False):
    return (probability(q,s,linear)+(probability(q,s,linear)-1)*np.ceil(-k/b))*(1-probability(q,s,linear))**(np.floor(k))

def dItemPb(q, s, k, b, question = False, linear = False):
    if k == 0:
        return probabilityDerivative(q, s, question, linear)
    return probabilityDerivative(q, s, question, linear)*(-1*(1-probability(q, s, linear))**(np.floor(k)-1))*(-1*(np.floor(k)+1)*np.floor(k/b)*(probability(q, s, linear)-1)+(np.floor(k)+1)*probability(q, s, linear)-1)

def calculateMarginal(position, data, estX, studentSize, linear = False):
    question = False if position < studentSize else True
    subData = [x for x in data if position == x[2]] if question else [x for x in data if position == x[1]]
    maxB = max([x[3] for x in subData])
    r = {}
    if not linear:
        r['Logistic at Mean'] = [ np.array([ probability(estX[x[2]], estX[x[1]], linear) for x in subData]).mean() ]
    for b in range(0, maxB+1):
        r["ACP k=" + str(b)] = [ np.array([ itemPb(estX[x[2]], estX[x[1]], b, x[3], linear) for x in subData ]).mean() ]
    for b in range(0, maxB+1):
        r["AME k=" + str(b)] = [ np.array([ dItemPb(estX[x[2]], estX[x[1]], b, x[3], question, linear) for x in subData ]).mean() ]
    return pd.DataFrame(r)

def calculateMarginals(data, estX, studentSize, linear = False): 
    studentResults = pd.concat([ calculateMarginal(x, data, estX, studentSize, linear) for x in range(0, studentSize)], ignore_index=True)
    rubricResults = pd.concat([ calculateMarginal(x, data, estX, studentSize, linear) for x in range(studentSize, len(estX))], ignore_index=True)
    return studentResults, rubricResults
