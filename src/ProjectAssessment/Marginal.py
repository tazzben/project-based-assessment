from statistics import mean
import numpy as np
import pandas as pd
from scipy.special import expit

def probability(q, s, xEst, iItem, linear = False):
    vS = np.dot(xEst, iItem) if len(xEst) > 0 else 0
    return q + s + vS if linear else expit(q + s + vS)

def mLogisticD(q, s, xEst, iItem, xPosition, question = False, student = False):
    probabilityWith = probability(q, s, xEst, iItem, False)
    xNew = xEst.copy()
    qN, sN = q, s
    if question:
        qN = 0
    elif student:
        sN = 0
    else:
        xNew[xPosition] = 0
    probabilityWithout = probability(qN, sN, xNew, iItem, False)
    return probabilityWith - probabilityWithout

def logisticD(q, s, xEst, iItem, question = False, student = False, xVal = 0):
    vS = np.dot(xEst, iItem) if len(xEst) > 0 else 0
    if question:
        m = q
    elif student:
        m = s
    else:
        m = xVal
    et = expit(q + s + vS)
    return m * et*(1 - et)

def linearD(q, s, question = False, student = False, xVal = 0):
    if question:
        return q
    if student:
        return s
    return xVal

def probabilityDerivative(q, s, xEst, iItem, question = False, student = False, linear = False, xVal = 0):
    return linearD(q, s, question, student, xVal) if linear else logisticD(q, s, xEst, iItem, question, student, xVal)

def kbcom(k, b):
    return -1 if k == b else 0

def itemPb(q, s, k, b, xEst, iItem, linear = False):
    pb = probability(q, s, xEst, iItem, linear)
    return (pb+(pb-1)*kbcom(k, b))*(1-pb)**(k)

def dItemPb(q, s, k, b, xEst, iItem, question = False, student = False, linear = False, xVal = 0):
    if k == 0:
        return probabilityDerivative(q, s, xEst, iItem, question, student, linear, xVal)
    pb = probability(q, s, xEst, iItem, linear)
    return probabilityDerivative(q, s, xEst, iItem, question, student, linear, xVal)*(-1*(1-pb)**(k-1))*(-1*(k+1)*kbcom(k, b)*(-1)*(pb-1)+(k+1)*pb-1)

def makeZero(estX, x, p, size):
    if size > 0:
        return estX[x[p]]
    return 0

def calculateMarginal(position, data, estX, studentSize, questionSize, nCol, linear = False):
    question = True if (position >= studentSize and position < (studentSize + questionSize)) else False
    student = True if position < studentSize else False
    xVarEst = estX[-nCol:] if nCol > 0 else []
    xVarPos = position - (studentSize + questionSize + nCol) if not (student or question) else 0
    xest = xVarEst[xVarPos] if len(xVarEst) > 0 else 0
    if student or question:
        subData = [x for x in data if position == x[2]] if question else [x for x in data if position == x[1]]
    else:
        subData = [x for x in data if x[xVarPos] != 0]
    minB = min((x[3] for x in subData))
    r = {}
    if not linear:
        r['Average Logistic'] = [ mean(( probability(makeZero(estX, x, 2, questionSize), makeZero(estX, x, 1, studentSize), xVarEst, x[-nCol:], linear) for x in subData)) ]
        r['Average Marginal Logistic'] = [ mean(( logisticD(makeZero(estX, x, 2, questionSize), makeZero(estX, x, 1, studentSize), xVarEst, x[-nCol:], question, student, xest) for x in subData)) ]
        r['Average Discrete Marginal Logistic'] = [ mean(( mLogisticD(makeZero(estX, x, 2, questionSize), makeZero(estX, x, 1, studentSize), xVarEst, x[-nCol:], xVarPos, question, student) for x in subData)) ]
    for b in range(0, minB+1):
        r["ACP k=" + str(b)] = [ mean(( itemPb(makeZero(estX, x, 2, questionSize), makeZero(estX, x, 1, studentSize), b, minB, xVarEst, x[-nCol:], linear) for x in subData )) ]
    for b in range(0, minB+1):
        r["AME k=" + str(b)] = [ mean(( dItemPb(makeZero(estX, x, 2, questionSize), makeZero(estX, x, 1, studentSize), b, minB, xVarEst, x[-nCol:], question, student, linear, xest) for x in subData )) ]
    return pd.DataFrame(r)

def calculateMarginals(data, estX, studentSize, questionSize, nCol, linear = False):
    if studentSize > 0:
        studentResults = pd.concat([ calculateMarginal(x, data, estX, studentSize, questionSize, nCol, linear) for x in range(0, studentSize)], ignore_index=True)
    else:
        studentResults = pd.DataFrame()
    if questionSize > 0:
        rubricResults = pd.concat([ calculateMarginal(x, data, estX, studentSize, questionSize, nCol, linear) for x in range(studentSize, studentSize + questionSize)], ignore_index=True)
    else:
        rubricResults = pd.DataFrame()
    if nCol > 0:
        varResults = pd.concat([ calculateMarginal(x, data, estX, studentSize, questionSize, nCol, linear) for x in range(studentSize + questionSize, len(estX))], ignore_index=True)
    else:
        varResults = pd.DataFrame()
    return studentResults, rubricResults, varResults
