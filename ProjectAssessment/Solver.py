from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import percentileofscore
from scipy.stats.distributions import chi2
from scipy.special import expit, xlog1py, xlogy
from tqdm import tqdm
from numba import njit
from .MakeTable import MakeTwoByTwoTable
from .Marginal import calculateMarginals

@njit
def kbcom(k, b):
    return -1 if k == b else 0

@njit
def itemPb2(q, s, k, b, xVari, itemi):
    if q == 0 and s == 0:
        return ( np.dot(np.ascontiguousarray(xVari), np.ascontiguousarray(itemi)), k, kbcom(k, b))
    if q == 0:
        return ( np.dot(np.ascontiguousarray(xVari), np.ascontiguousarray(itemi)) + s, k, kbcom(k, b))
    if s == 0:
        return ( np.dot(np.ascontiguousarray(xVari), np.ascontiguousarray(itemi)) + q, k, kbcom(k, b))
    return ( np.dot(np.ascontiguousarray(xVari), np.ascontiguousarray(itemi)) + q + s, k, kbcom(k, b))

@njit
def itemPb(q, s, k, b):
    return ( q + s, k, kbcom(k, b))

@njit
def itemLoop2(x, data, cols = 0, no_q = False, no_s = False):
    if no_q and no_s:
        return [ itemPb2(0, 0, item[0], item[3], x[-cols:], item[-cols:]) for item in data ]
    if no_q:
        return [ itemPb2(0, x[int(item[1])], item[0], item[3], x[-cols:], item[-cols:]) for item in data ]
    if no_s:
        return [ itemPb2(x[int(item[2])], 0, item[0], item[3], x[-cols:], item[-cols:]) for item in data ]
    return [ itemPb2(x[int(item[2])], x[int(item[1])], item[0], item[3], x[-cols:], item[-cols:]) for item in data ]

@njit
def itemLoop(x, data):
    return [ itemPb(x[int(item[2])], x[int(item[1])], item[0], item[3]) for item in data ]

@njit
def itemLoopRestricted(x, data):
    return [ ( x[0], item[0], kbcom(item[0], item[3])) for item in data ]

def opFunction(x, data, linear = False, cols = 0, no_q = False, no_s = False):
    if cols > 0:
        dem = np.array(itemLoop2(x, data, cols, no_q, no_s))
    else:
        dem = np.array(itemLoop(x, data))
    vS = dem[:,0] if linear else expit(dem[:,0])
    return -np.sum(xlogy(1,  vS + (vS - 1) * dem[:,2]) + xlog1py(dem[:,1], -vS))

def opRestricted (x, data, linear = False):
    dem = np.array(itemLoopRestricted(x, data))
    vS = dem[:,0] if linear else expit(dem[:,0])
    return -np.sum(xlogy(1,  vS + (vS - 1) * dem[:,2]) + xlog1py(dem[:,1], -vS))

def solve(dataset, summary = True, linear = False, columns = None, no_q = False, no_s = False):
    studentCode, uniqueStudents = pd.factorize(dataset['student'])
    questionCode, uniqueQuestion = pd.factorize(dataset['rubric'])
    if no_q and no_s:
        smap = []
    elif no_q:
        smap = uniqueStudents.tolist()
    elif no_s:
        smap = uniqueQuestion.tolist()
    else:
        questionCode = questionCode + uniqueStudents.size
        smap = np.concatenate((uniqueStudents, uniqueQuestion), axis=None).tolist()
    data = list(zip(dataset['k'].to_numpy().flatten().tolist(), studentCode.tolist(), questionCode.tolist(), dataset['bound'].to_numpy().flatten().tolist()))
    for i, _ in enumerate(data):
        for col in columns:
            data[i] = data[i] + (dataset[col].iloc[i],)
    if linear:
        if no_q and no_s:
            bounds = ((0, 1),) * len(columns)
        elif no_q:
            bounds = ((0, 1),) * (uniqueStudents.size + len(columns))
        elif no_s:
            bounds = ((0, 1),) * (uniqueQuestion.size + len(columns))
        else:
            bounds = ((0, 1),) * (uniqueStudents.size + uniqueQuestion.size + len(columns))
    else:
        bounds = None
    minValue = minimize(opFunction, np.array((1/(2*(1+dataset['k'].mean())),)*(len(smap) + len(columns)), np.dtype(float)), args=(np.array(data, np.dtype(float)), linear, len(columns), no_q, no_s), method='Powell', bounds=bounds)
    if minValue.success:
        estX = minValue.x.flatten().tolist()
        varNames = smap + columns
        fullResults = list(zip(varNames, estX))
        cols = ['Variable', 'Value']
        if no_q and no_s:
            studentResults = pd.DataFrame(columns=cols)
            questionResults = pd.DataFrame(columns=cols)
            varResults = pd.DataFrame(fullResults, columns=cols)
        elif no_q:
            studentResults = pd.DataFrame(fullResults[:uniqueStudents.size], columns=cols)
            questionResults = pd.DataFrame(columns=cols)
            varResults = pd.DataFrame(fullResults[uniqueStudents.size:], columns=cols)
        elif no_s:
            studentResults = pd.DataFrame(columns=cols)
            questionResults = pd.DataFrame(fullResults[:uniqueQuestion.size], columns=cols)
            varResults = pd.DataFrame(fullResults[uniqueQuestion.size:], columns=cols)
        else:
            studentResults = pd.DataFrame(fullResults[:uniqueStudents.size], columns=cols)
            questionResults = pd.DataFrame(fullResults[uniqueStudents.size:len(smap)], columns=cols)
            varResults = pd.DataFrame(fullResults[len(smap):], columns=cols)
        if not summary:
            return {
                'student': studentResults,
                'rubric': questionResults,
                'variables': varResults
            }
        parStudents, parQuestion, parVar = (len(studentResults.index), len(questionResults.index), len(varResults.index))
        studentMarginals, rubricMarginals, varMarginals = calculateMarginals(data, estX, parStudents, parQuestion, parVar, linear)
        d = {
            'student': studentResults.join(studentMarginals),
            'rubric': questionResults.join(rubricMarginals),
            'variables': varResults.join(varMarginals)
        }
        d['LogLikelihood'] = -1.0 * minValue.fun
        d['AIC'] = 2*(parStudents + parQuestion + parVar)+2*minValue.fun
        d['BIC'] = (parStudents + parQuestion + parVar)*np.log(len(studentCode))+2*minValue.fun
        d['n'] = len(studentCode)
        d['NumberOfParameters'] = parStudents + parQuestion + parVar
        if linear:
            boundsSingle = ((0, 1),)
        else:
            boundsSingle = None
        minRestricted = minimize(opRestricted, np.array((1/(1+dataset['k'].mean()),), np.dtype(float)), args=(np.array(data, np.dtype(float)), linear), method='Powell', bounds=boundsSingle)
        if minRestricted.success:
            d['McFadden'] = 1 - minValue.fun/minRestricted.fun
            d['LR'] = -2*(-1*minRestricted.fun+minValue.fun)
            d['Chi-Squared'] = chi2.sf(d['LR'], (parStudents + parQuestion + parVar - 1))
        return d
    if not summary:
        return None
    raise Exception(minValue.message)

def bootstrapRow (dataset, columns, rubric=False, linear=False, no_q = False, no_s = False):
    key = 'rubric' if rubric else 'student'
    ids = dataset[key].unique().flatten().tolist()
    randomGroupIds = np.random.choice(ids, size=len(ids), replace=True)
    l = []
    for c, i in enumerate(randomGroupIds):
        rows = dataset[dataset[key]==i]
        rows = rows.assign(rubric=c) if rubric else rows.assign(student=c)
        l.append(rows)
    resultData = pd.concat(l, ignore_index=True)
    return solve(resultData, False, linear, columns, no_q, no_s)

def CallRow(row):
    key = 'student' if row['rubric'] else 'rubric'
    r = bootstrapRow(row['dataset'], row['columns'], row['rubric'], row['linear'], row['no_q'], row['no_s'])
    if r is not None:
        return (r[key], r['variables'])
    return None

def bootstrap(dataset, n, rubric=False, linear=False, columns=None, no_q = False, no_s = False):
    l = []
    rows = [
        {
            'dataset': dataset,
            'rubric': rubric,
            'linear': linear,
            'columns': columns,
            'no_q': no_q,
            'no_s': no_s
        }
    ]*n
    p = Pool()
    nones = []
    for _, result in tqdm(enumerate(p.imap_unordered(CallRow, rows)), total=n):
        if result is not None:
            keyresult, varresult = result
            l.append(keyresult)
            l.append(varresult)
        else:
            nones.append(1)
    p.close()
    p.join()
    return {
        'results': pd.concat(l, ignore_index=True),
        'nones': len(nones)
    }

def compareKBound(x):
    return pd.to_numeric(x, downcast='integer')

def getResults(dataset: pd.DataFrame,c=0.025, rubric=False, n=1000, linear=False, columns=None, no_students=False, no_questions=False):
    """
    Estimates the parameters of the model and produces confidence intervals for the estimates using a bootstrap method.

    Parameters:
    -----------
    dataset : Pandas DataFrame
        A dataframe with the following columns: k, student, rubric, bound.  Student and rubric identifiers for the student and rubric items.  Bound is maximum bin in the rubric that can be reached.  k is the bin the student reached on a given rubric item.
    c : float
        Confidence level for the confidence intervals.  Defaults to 0.025.
    rubric : bool
        Switches the bootstrap to treating the rubric rows as blocks instead of the students.  Defaults to False.
    n : int
        Number of iterations for the bootstrap.  Defaults to 1000.
    linear: bool
        Uses a simple linear combination of the rubric and student items instead of a sigmoid function.  Defaults to False.
    columns: list
        A list of column names to include in the model.  The column names cannot be in common with any of the rubric row identifiers.  Defaults to None.

    Returns:
        Tuple:
            Rubric and Arbitrary Column Estimates : Pandas DataFrame
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
            Log Likelihood : float

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
    columns = [c.strip() for c in columns] if isinstance(columns, (list, tuple)) else []
    if len(columns) > 0 and not set(columns).issubset(dataset.columns):
        raise Exception('Specified columns not in dataset')
    dataset.dropna(inplace=True)
    dataset = dataset[pd.to_numeric(dataset['k'], errors='coerce').notnull()]
    dataset = dataset[pd.to_numeric(dataset['bound'], errors='coerce').notnull()]
    dataset[["k", "bound"]] = dataset[["k", "bound"]].apply(compareKBound)
    dataset = dataset[dataset['k'] <= dataset['bound']]
    no_s = False
    no_q = False
    if len(columns) > 0:
        for spec in columns:
            dataset = dataset[pd.to_numeric(dataset[spec], errors='coerce').notnull()]
        dataset[columns] = dataset[columns].apply(pd.to_numeric)
        if set(columns).intersection(set(dataset['rubric'].unique())):
            raise Exception('Specified columns cannot be in common with any of the rubric row identifiers.')
        if no_students:
            no_s = True
        if no_questions:
            no_q = True
    if not len(dataset.index) > 0:
        raise Exception('Invalid pandas dataset, empty dataset.')
    print("Estimating Parameters ...")
    estimates = solve(dataset, linear=linear, columns=columns, no_q=no_q, no_s=no_s)
    print("... Done. Bootstrapping ...")
    if estimates is not None:
        results = bootstrap(dataset, n, rubric, linear=linear, columns=columns)
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
        combined = pd.concat([estimates['rubric'], estimates['variables']], ignore_index=True)
        return (combined, estimates['student'], pd.DataFrame(l), results['nones'], estimates['n'], estimates['NumberOfParameters'], estimates['AIC'], estimates['BIC'], McFadden, LR, ChiSquared, estimates['LogLikelihood'])
    else:
        raise Exception('Could not find estimates.')

def DisplayResults(dataset: pd.DataFrame,c=0.025, rubric=False, n=1000, linear=False, columns=None, no_students=False, no_questions=False):
    """
    Estimates the parameters of the model and produces confidence intervals for the estimates using a bootstrap method. Results are printed out to the console.

    Parameters:
    -----------
    dataset : Pandas DataFrame
        A dataframe with the following columns: k, student, rubric, bound.  Student and rubric identifiers for the student and rubric items.  Bound is maximum bin in the rubric that can be reached.  k is the bin the student reached on a given rubric item.
    c : float
        Confidence level for the confidence intervals.  Defaults to 0.025.
    rubric : bool
        Switches the bootstrap to treating the rubric rows as blocks instead of the students.  Defaults to False.
    n : int
        Number of iterations for the bootstrap.  Defaults to 1000.
    linear: bool
        Uses a simple linear combination of the rubric and student items instead of a sigmoid function.  Defaults to False.
    columns: list
        A list of column names to include in the model. The column names cannot be in common with any of the rubric row identifiers. Defaults to None.

    Returns:
        Tuple:
            Rubric and Arbitrary Column Estimates : Pandas DataFrame
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
            Log Likelihood : float

    """
    rubricR, studentR, bootstrapR, countE, obs, param, AIC, BIC, McFadden, LR, ChiSquared, LogLikelihood = getResults(dataset, c, rubric, n, linear, columns=columns, no_students=no_students, no_questions=no_questions)
    warnings = []
    if rubric is True:
        printedStudent = studentR.merge(bootstrapR, on='Variable', how='left')
        if isinstance(columns, list) and len(columns) > 0:
            specialbootstrap = bootstrapR[bootstrapR['Variable'].isin(columns)]
            printedRubric = rubricR.merge(specialbootstrap, on='Variable', how='left')
        else:
            printedRubric = rubricR
    else:
        printedRubric = rubricR.merge(bootstrapR, on='Variable', how='left')
        printedStudent = studentR
    addedMes = ""
    if isinstance(columns, list) and len(columns) > 0:
        addedMes = " and Arbitrary Columns"
    print('Rubric' + addedMes + ' Estimates:')
    print(printedRubric)
    print('Student Estimates:')
    print(printedStudent)
    if countE > 0:
        warnings.append(str(countE) + ' of ' + str(n) + ' bootstrap samples were empty.')

    x = []

    x.append(["Number of Observations", obs])
    x.append(["Number of Parameters", param])
    x.append(["Log Likelihood", LogLikelihood])
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
    return (rubricR, studentR, bootstrapR, countE, obs, param, AIC, BIC, McFadden, LR, ChiSquared, LogLikelihood)

def SaveResults(dataset: pd.DataFrame,c=0.025, rubric=False, n=1000, linear=False, rubricFile = 'rubric.csv', studentFile = 'student.csv', outputFile = 'output.csv', columns=None, no_students=False, no_questions=False):
    """
    Estimates the parameters of the model and produces confidence intervals for the estimates using a bootstrap method. Results are printed out to the console and saved to CSV files.

    Parameters:
    -----------
    dataset : Pandas DataFrame
        A dataframe with the following columns: k, student, rubric, bound.  Student and rubric identifiers for the student and rubric items.  Bound is maximum bin in the rubric that can be reached.  k is the bin the student reached on a given rubric item.
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
    columns: list
        A list of column names to include in the model. The column names cannot be in common with any of the rubric row identifiers. Defaults to None.

    Returns:
        Tuple:
            Rubric and Arbitrary Column Estimates : Pandas DataFrame
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
            Log Likelihood : float

    """
    rubricR, studentR, bootstrapR, countE, obs, param, AIC, BIC, McFadden, LR, ChiSquared, LogLikelihood  = DisplayResults(dataset, c, rubric, n, linear, columns=columns, no_students=no_students, no_questions=no_questions)

    if rubric is True:
        printedStudent = studentR.merge(bootstrapR, on='Variable', how='left')
        if isinstance(columns, list) and len(columns) > 0:
            specialbootstrap = bootstrapR[bootstrapR['Variable'].isin(columns)]
            printedRubric = rubricR.merge(specialbootstrap, on='Variable', how='left')
        else:
            printedRubric = rubricR
    else:
        printedRubric = rubricR.merge(bootstrapR, on='Variable', how='left')
        printedStudent = studentR

    printedRubric.to_csv(rubricFile, index=False)
    printedStudent.to_csv(studentFile, index=False)
    output = {
        'Number of Observations': obs,
        'Number of Parameters': param,
        'Log Likelihood': LogLikelihood,
        'AIC': AIC,
        'BIC': BIC,
        'McFadden R^2': McFadden,
        'Likelihood Ratio Test Statistic': LR,
        'Chi-Squared LR Test P-Value': ChiSquared,
    }
    pd.DataFrame.from_dict(output, orient='index').to_csv(outputFile, header=False)
    return (rubricR, studentR, bootstrapR, countE, obs, param, AIC, BIC, McFadden, LR, ChiSquared, LogLikelihood)
