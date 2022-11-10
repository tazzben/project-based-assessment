import numpy as np
import pandas as pd
from scipy.special import expit

def probability(q, s, xEst, iItem, linear = False):
    vS = np.dot(xEst, iItem)
    return q + s + vS if linear else expit(q + s + vS)

def logisticD(q, s, xEst, iItem, question = False, student = False, xVal = 0):
    vS = np.dot(xEst, iItem)
    if question:
        m = q
    elif student:
        m = s
    else:
        m = xVal
    return m * expit(q + s + vS)*(1 - expit(q + s + vS))

def linearD(q, s, question = False, student = False, xVal = 0):
    if question:
        return q
    if student:
        return s
    return xVal

def probabilityDerivative(q, s, xEst, iItem, question = False, student = False, linear = False, xVal = 0):
    return linearD(q, s, question, student, xVal) if linear else logisticD(q, s, xEst, iItem, question, student, xVal)

def itemPb(q, s, k, b, xEst, iItem, linear = False):
    return (probability(q, s, xEst, iItem, linear)+(probability(q, s, xEst, iItem, linear)-1)*np.ceil(-k/b))*(1-probability(q, s, xEst, iItem, linear))**(np.floor(k))

def dItemPb(q, s, k, b, xEst, iItem, question = False, student = False, linear = False, xVal = 0):
    if k == 0:
        return probabilityDerivative(q, s, xEst, iItem, question, student, linear, xVal)
    return probabilityDerivative(q, s, xEst, iItem, question, student, linear, xVal)*(-1*(1-probability(q, s, xEst, iItem, linear))**(np.floor(k)-1))*(-1*(np.floor(k)+1)*np.floor(k/b)*(probability(q, s, xEst, iItem, linear)-1)+(np.floor(k)+1)*probability(q, s, xEst, iItem, linear)-1)

def calculateMarginal(position, data, estX, studentSize, questionSize, nCol, linear = False):
    question = True if (position >= studentSize and position < (studentSize + questionSize)) else False
    student = True if position < studentSize else False
    if nCol > 0:
        xVarEst = estX[-nCol:]
    else:
        xVarEst = []
    if student or question:
        xVarPos = 0
        subData = [x for x in data if position == x[2]] if question else [x for x in data if position == x[1]]
    else:
        xVarPos = (studentSize + questionSize) - position - 1
        subData = [x for x in data if x[xVarPos] != 0]
    xest = xVarEst[xVarPos] if len(xVarEst) > 0 else 0
    minB = min([x[3] for x in subData])
    r = {}
    if not linear:
        r['Average Logistic'] = [ np.array([ probability(estX[x[2]], estX[x[1]], xVarEst, x[-nCol:], linear) for x in subData]).mean() ]
        r['Average Marginal Logistic'] = [ np.array([ logisticD(estX[x[2]], estX[x[1]], xVarEst, x[-nCol:], question, student, xest) for x in subData]).mean() ]
    for b in range(0, minB+1):
        r["ACP k=" + str(b)] = [ np.array([ itemPb(estX[x[2]], estX[x[1]], b, minB, xVarEst, x[-nCol:], linear) for x in subData ]).mean() ]
    for b in range(0, minB+1):
        r["AME k=" + str(b)] = [ np.array([ dItemPb(estX[x[2]], estX[x[1]], b, minB, xVarEst, x[-nCol:], question, student, linear, xest) for x in subData ]).mean() ]
    return pd.DataFrame(r)

def calculateMarginals(data, estX, studentSize, questionSize, nCol, linear = False):
    studentResults = pd.concat([ calculateMarginal(x, data, estX, studentSize, questionSize, nCol, linear) for x in range(0, studentSize)], ignore_index=True)
    rubricResults = pd.concat([ calculateMarginal(x, data, estX, studentSize, questionSize, nCol, linear) for x in range(studentSize, studentSize + questionSize)], ignore_index=True)
    if nCol > 0:
        varResults = pd.concat([ calculateMarginal(x, data, estX, studentSize, questionSize, nCol, linear) for x in range(studentSize + questionSize, len(estX))], ignore_index=True)
    else:
        varResults = pd.DataFrame()
    return studentResults, rubricResults, varResults
