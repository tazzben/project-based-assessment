import math
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import percentileofscore
from scipy.stats.distributions import chi2
from scipy.special import expit, xlog1py, xlogy
from progress.bar import Bar
from .MakeTable import MakeTwoByTwoTable
from .Marginal import calculateMarginals

def itemPb(q, s, k, b, linear = False):
    if linear:
        return xlogy(1, q + s + (q + s - 1) * np.ceil(-k/b)) + xlog1py(np.floor(k), -1*(q+s))
    return  xlogy(1, expit(q+s) + (expit(q+s) - 1) * np.ceil(-k/b)) + xlog1py(np.floor(k), -1*expit(q+s))

def opFunction(x, data, linear = False):
    return -1.0 * (np.array([ itemPb(x[item[2]], x[item[1]], item[0], item[3], linear) for item in data ]).sum())

def opRestricted (x, data, linear = False):
    return -1.0 * (np.array([ itemPb(x[0], 0, item[0], item[3], linear) for item in data ]).sum())

def solve(dataset, summary = True, linear = False):
    studentCode, uniqueStudents = pd.factorize(dataset['student'])
    questionCode, uniqueQuestion = pd.factorize(dataset['rubric'])
    questionCode = questionCode + uniqueStudents.size
    smap = np.concatenate((uniqueStudents, uniqueQuestion), axis=None).tolist()
    data = list(zip(dataset['k'].to_numpy().flatten().tolist(), studentCode.tolist(), questionCode.tolist(), dataset['bound'].to_numpy().flatten().tolist()))
    if linear:
        bounds = [(0, 1)] * (uniqueStudents.size + uniqueQuestion.size)
    else:
        bounds = None
    minValue = minimize(opFunction, [1/(2*(1+dataset['k'].mean()))]*len(smap), args=(data, linear), method='Powell', bounds=bounds)
    if minValue.success:
        estX = minValue.x.flatten().tolist()
        fullResults = list(zip(smap, estX))
        cols = ['Variable', 'Value']
        studentResults = pd.DataFrame(fullResults[:uniqueStudents.size], columns=cols)
        questionResults = pd.DataFrame(fullResults[uniqueStudents.size:], columns=cols)
        if not summary:
            return {
                'student': studentResults,
                'rubric': questionResults
            }
        studentMarginals, rubricMarginals = calculateMarginals(data, estX, uniqueStudents.size, linear)
        d = {
            'student': studentResults.join(studentMarginals),
            'rubric': questionResults.join(rubricMarginals)
        }
        d['AIC'] = 2*(uniqueStudents.size+uniqueQuestion.size)+2*minValue.fun
        d['BIC'] = (uniqueStudents.size+uniqueQuestion.size)*math.log(len(studentCode))+2*minValue.fun
        d['n'] = len(studentCode)
        d['NumberOfParameters'] = uniqueStudents.size+uniqueQuestion.size
        if linear:
            boundsSingle = [(0, 1)]
        else:
            boundsSingle = None
        minRestricted = minimize(opRestricted, [1/(1+dataset['k'].mean()),], args=(data, linear), method='Powell', bounds=boundsSingle)
        if minRestricted.success:
            d['McFadden'] = 1 - minValue.fun/minRestricted.fun
            d['LR'] = -2*(-1*minRestricted.fun+minValue.fun)
            d['Chi-Squared'] = chi2.sf(d['LR'], (uniqueStudents.size+uniqueQuestion.size-1))
        return d
    if not summary:
        return None
    raise Exception(minValue.message)

def bootstrapRow (dataset, rubric=False, linear=False):
    key = 'rubric' if rubric else 'student'
    ids = dataset[key].unique().flatten().tolist()
    randomGroupIds = np.random.choice(ids, size=len(ids), replace=True)
    l = []
    for c, i in enumerate(randomGroupIds):
        rows = dataset[dataset[key]==i]
        rows = rows.assign(rubric=c) if rubric else rows.assign(student=c)
        l.append(rows)
    resultData = pd.concat(l, ignore_index=True)
    return solve(resultData, False, linear)

def CallRow(row):
    key = 'student' if row['rubric'] else 'rubric'
    r = bootstrapRow(row['dataset'], row['rubric'], row['linear'])
    if r is not None:
        return r[key]
    return None

def bootstrap(dataset, n, rubric=False, linear=False):
    l = []
    rows = [
        {
            'dataset': dataset,
            'rubric': rubric,
            'linear': linear
        }
    ]*n
    b = Bar('Processing', max=n)
    p = Pool()
    nones = []
    for _, result in enumerate(p.imap_unordered(CallRow, rows)):
        if result is not None:
            l.append(result)
        else:
            nones.append(1)
        b.next()
    p.close()
    p.join()
    b.finish()
    return {
        'results': pd.concat(l, ignore_index=True),
        'nones': len(nones)
    }

def getResults(dataset: pd.DataFrame,c=0.025, rubric=False, n=1000, linear=False):
    """
    Estimates the parameters of the model and produces confidence intervals for the estimates using a bootstrap method.

    Parameters:
    -----------
    dataset : Pandas DataFrame
        A dataframe with the following columns: k, student, rubric, bound.  Student and rubric identifiers for the student and rubric items.  Bound is maximum bin in the rubric that can be reached.  k is the bin the bin the student reached on a given rubric item.
    c : float
        Confidence level for the confidence intervals.  Defaults to 0.025.
    rubric : bool
        Switches the bootstrap to treating the rubric rows as blocks instead of the students.  Defaults to False.
    n : int
        Number of iterations for the bootstrap.  Defaults to 1000.
    linear: bool
        Uses a simple linear combination of the rubric and student items instead of a sigmoid function.  Defaults to False.


    Returns:
        Tuple:
            Rubric Estimates : Pandas DataFrame
            Student Estimates : Pandas DataFrame
            Bootstrap CIs and P-Values : Pandas DataFrame
            # of Times no solution could be found in the Bootstrap : int
            Number of Observations : int
            Number of Parameters : int
            AIC : float
            BIC : float
            McFadden R^2 : float
            Likelihood Ratio Test Statistic of the model : float
            Chi-Squared P-Value of the model : float

    """
    if not isinstance(dataset, pd.DataFrame):
        raise Exception('dataset must be a Pandas DataFrame')
    dataset = dataset.rename(columns=lambda x: x.strip())
    if not set(['k','bound', 'student', 'rubric']).issubset(dataset.columns):
        raise Exception('Invalid pandas dataset, missing columns. k, bound, student, and rubric are required.')
    if not isinstance(c, float) and c >= 0 and c <= 0.5:
        raise Exception('c must be a float between 0 and 0.5')
    if not isinstance(rubric, bool):
        raise Exception('rubric must be a boolean')
    if not isinstance(n, int) and n > 0:
        raise Exception('n must be an integer greater than 0')
    dataset.dropna(inplace=True)
    dataset = dataset[pd.to_numeric(dataset['k'], errors='coerce').notnull()]
    dataset = dataset[pd.to_numeric(dataset['bound'], errors='coerce').notnull()]
    if not len(dataset.index) > 0:
        raise Exception('Invalid pandas dataset, empty dataset.')
    estimates = solve(dataset, linear=linear)
    if estimates is not None:
        results = bootstrap(dataset, n, rubric, linear=linear)
        r = results['results']
        l = []
        for var in r['Variable'].unique():
            df = r[r['Variable'] == var]['Value']
            ci = (df.quantile(q=c), df.quantile(q=(1-c)))
            vDict = {
                'Variable': var,
                'Confidence Interval': ci,
            }
            if not linear:
                pvalue = (percentileofscore(df, 0) / 100)*2 if np.mean(df) > 0 else (1 - (percentileofscore(df, 0) / 100))*2
                vDict['P-Value'] = pvalue
            l.append(vDict)
        McFadden = None
        LR = None
        ChiSquared = None
        if "McFadden" in estimates:
            McFadden = estimates["McFadden"]
            LR = estimates["LR"]
            ChiSquared = estimates["Chi-Squared"]
        return (estimates['rubric'], estimates['student'], pd.DataFrame(l), results['nones'], estimates['n'], estimates['NumberOfParameters'], estimates['AIC'], estimates['BIC'], McFadden, LR, ChiSquared)
    else:
        raise Exception('Could not find estimates.')

def DisplayResults(dataset: pd.DataFrame,c=0.025, rubric=False, n=1000, linear=False):
    """
    Estimates the parameters of the model and produces confidence intervals for the estimates using a bootstrap method. Results are printed out to the console.

    Parameters:
    -----------
    dataset : Pandas DataFrame
        A dataframe with the following columns: k, student, rubric, bound.  Student and rubric identifiers for the student and rubric items.  Bound is maximum bin in the rubric that can be reached.  k is the bin the bin the student reached on a given rubric item.
    c : float
        Confidence level for the confidence intervals.  Defaults to 0.025.
    rubric : bool
        Switches the bootstrap to treating the rubric rows as blocks instead of the students.  Defaults to False.
    n : int
        Number of iterations for the bootstrap.  Defaults to 1000.
    linear: bool
        Uses a simple linear combination of the rubric and student items instead of a sigmoid function.  Defaults to False.


    Returns:
        Tuple:
            Rubric Estimates : Pandas DataFrame
            Student Estimates : Pandas DataFrame
            Bootstrap CIs and P-Values : Pandas DataFrame
            # of Times no solution could be found in the Bootstrap : int
            Number of Observations : int
            Number of Parameters : int
            AIC : float
            BIC : float
            McFadden R^2 : float
            Likelihood Ratio Test Statistic of the model : float
            Chi-Squared P-Value of the model : float

    """
    rubricR, studentR, bootstrapR, countE, obs, param, AIC, BIC, McFadden, LR, ChiSquared = getResults(dataset, c, rubric, n, linear)
    warnings = []
    if rubric is True:
        printedStudent = studentR.merge(bootstrapR, on='Variable', how='inner')
        printedRubric = rubricR
    else:
        printedRubric = rubricR.merge(bootstrapR, on='Variable', how='inner')
        printedStudent = studentR
    print('Rubric Estimates:')
    print(printedRubric)
    print('Student Estimates:')
    print(printedStudent)
    if countE > 0:
        warnings.append(str(countE) + ' of ' + str(n) + ' bootstrap samples were empty.')

    x = []

    x.append(["Number of Observations", obs])
    x.append(["Number of Parameters", param])
    x.append(["AIC", AIC])
    x.append(["BIC", BIC])
    if McFadden is not None:
        x.append(["McFadden R^2", McFadden])
        x.append(["Likelihood Ratio Test Statistic", LR])
        x.append(["Chi-Squared LR P-Value", ChiSquared])
    else:
        warnings.append('McFadden R^2, Likelihood Ratio Test, and Chi-Squared LR Test could not be displayed because the restricted model could not be solved.')
    MakeTwoByTwoTable(x)
    if len(warnings) > 0:
        print('Warnings:')
        for warning in warnings:
            print(warning)
    return (rubricR, studentR, bootstrapR, countE, obs, param, AIC, BIC, McFadden, LR, ChiSquared)

def SaveResults(dataset: pd.DataFrame,c=0.025, rubric=False, n=1000, linear=False, rubricFile = 'rubric.csv', studentFile = 'student.csv', outputFile = 'output.csv'):
    """
    Estimates the parameters of the model and produces confidence intervals for the estimates using a bootstrap method. Results are printed out to the console and saved to CSV files.

    Parameters:
    -----------
    dataset : Pandas DataFrame
        A dataframe with the following columns: k, student, rubric, bound.  Student and rubric identifiers for the student and rubric items.  Bound is maximum bin in the rubric that can be reached.  k is the bin the bin the student reached on a given rubric item.
    c : float
        Confidence level for the confidence intervals.  Defaults to 0.025.
    rubric : bool
        Switches the bootstrap to treating the rubric rows as blocks instead of the students.  Defaults to False.
    n : int
        Number of iterations for the bootstrap.  Defaults to 1000.
    linear: bool
        Uses a simple linear combination of the rubric and student items instead of a sigmoid function.  Defaults to False.
    rubricFile : str
        File name/path for the rubric results.  Defaults to 'rubric.csv'.
    studentFile : str
        File name/path for the student results.  Defaults to 'student.csv'.
    outputFile : str
        File name/path for the summary output results.  Defaults to 'output.csv'.

    Returns:
        Tuple:
            Rubric Estimates : Pandas DataFrame
            Student Estimates : Pandas DataFrame
            Bootstrap CIs and P-Values : Pandas DataFrame
            # of Times no solution could be found in the Bootstrap : int
            Number of Observations : int
            Number of Parameters : int
            AIC : float
            BIC : float
            McFadden R^2 : float
            Likelihood Ratio Test Statistic of the model : float
            Chi-Squared P-Value of the model : float

    """
    rubricR, studentR, bootstrapR, countE, obs, param, AIC, BIC, McFadden, LR, ChiSquared  = DisplayResults(dataset, c, rubric, n, linear)
    if rubric is True:
        printedStudent = studentR.merge(bootstrapR, on='Variable', how='inner')
        printedRubric = rubricR
    else:
        printedRubric = rubricR.merge(bootstrapR, on='Variable', how='inner')
        printedStudent = studentR
    printedRubric.to_csv(rubricFile, index=False)
    printedStudent.to_csv(studentFile, index=False)
    output = {
        'Number of Observations': obs,
        'Number of Parameters': param,
        'AIC': AIC,
        'BIC': BIC,
        'McFadden R^2': McFadden,
        'Likelihood Ratio Test Statistic': LR,
        'Chi-Squared LR Test P-Value': ChiSquared,
    }
    pd.DataFrame.from_dict(output, orient='index').to_csv(outputFile, header=False)
    return (rubricR, studentR, bootstrapR, countE, obs, param, AIC, BIC, McFadden, LR, ChiSquared)
