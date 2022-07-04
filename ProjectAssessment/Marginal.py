import numpy as np
import pandas as pd
from scipy.special import expit

def probability(q, s, linear = False):
    if linear:
        return q + s
    return expit(q + s)

def logisticD(q, s, question = False):
    return q * np.exp(q+s) / (1 + np.exp(q+s))**2 if question else s * np.exp(q+s) / (1 + np.exp(q+s))

def linearD(q, s, question = False):
    return q if question else s

def probabilityDerivative(q, s, question = False, linear = False):
    if linear:
        return linearD(q, s, question)
    return logisticD(q, s, question)

def dItemPb(q, s, k, b, question = False, linear = False):
    return -np.floor(k) * probabilityDerivative(q, s, question, linear) / (1 - probability(q, s, linear)) + (probabilityDerivative(q, s, question, linear) + probabilityDerivative(q, s, question, linear) * np.ceil(-k/b))/(probability(q, s, linear) + (probability(q, s, linear) - 1) * np.ceil(-k/b))
