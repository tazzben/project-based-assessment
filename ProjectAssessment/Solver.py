import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
from scipy import stats
from scipy.stats.distributions import chi2
from scipy.special import expit, xlog1py
from multiprocessing import Pool
from progress.bar import Bar
from prettytable import PrettyTable

def logistic(q, s):
	return expit(q+s)

def itemPb(q, s, k, b):
	return np.log(logistic(q,s) + (logistic(q,s) - 1) * np.ceil(-k/b)) + xlog1py(np.floor(k), -1*logistic(q,s)) 

def opFunction(x, data):
	array = np.array([ itemPb(x[item[1]], x[item[2]], item[0], item[3]) for item in data ])
	return -1.0 * (array.sum())

def opRestricted (x, data):
	array = np.array([ itemPb(x[0], 0, item[0], item[3]) for item in data ])
	return -1.0 * (array.sum())

def solve(dataset, summary = True):
	studentCode, uniqueStudents = pd.factorize(dataset['student'])
	questionCode, uniqueQuestion = pd.factorize(dataset['rubric'])
	questionCode = questionCode + uniqueStudents.size
	smap = np.concatenate((uniqueStudents, uniqueQuestion), axis=None).tolist()
	data = list(zip(dataset['k'].to_numpy().flatten().tolist(), studentCode.tolist(), questionCode.tolist(), dataset['bound'].to_numpy().flatten().tolist()))
	minValue = minimize(opFunction, [1/(2*(1+dataset['k'].mean()))]*len(smap), args=(data, ), method='Powell')
	if (minValue.success):
		lmap = map(lambda x: logistic(x, 0), minValue.x.flatten().tolist())
		fullResults = list(zip(smap, minValue.x.flatten().tolist(), lmap))
		studentResults = fullResults[:uniqueStudents.size]
		questionResults = fullResults[uniqueStudents.size:]
		d = {
			'student': pd.DataFrame(studentResults, columns=['Variable', 'Value', 'Logistic Transformed Value']),
			'rubric': pd.DataFrame(questionResults, columns=['Variable', 'Value', 'Logistic Transformed Value'])
		}
		if not summary:
			return d
		d['AIC'] = 2*(uniqueStudents.size+uniqueQuestion.size)+2*minValue.fun
		d['BIC'] = (uniqueStudents.size+uniqueQuestion.size)*math.log(len(studentCode))+2*minValue.fun
		d['n'] = len(studentCode)
		d['NumberOfParameters'] = uniqueStudents.size+uniqueQuestion.size
		minRestricted = minimize(opRestricted, [1/(1+dataset['k'].mean()),], args=(data, ), method='Powell')
		if (minRestricted.success):
			d['McFadden'] = 1 - minValue.fun/minRestricted.fun
			d['LR'] = -2*math.log(math.exp(-1*minRestricted.fun+minValue.fun))
			d['Chi-Squared'] = chi2.sf(d['LR'], (uniqueStudents.size+uniqueQuestion.size-1))
		return d
	else:
		raise Exception(minValue.message)

def bootstrapRow (dataset, rubric=False):
	key = 'rubric' if rubric else 'student'
	ids = dataset[key].unique().flatten().tolist()
	randomGroupIds = np.random.choice(ids, size=len(ids), replace=True)
	l = []
	for c, i in enumerate(randomGroupIds):
		rows = dataset[dataset[key]==i]
		rows.assign(rubric=c) if rubric else rows.assign(student=c)
		l.append(rows)
	resultData = pd.concat(l, ignore_index=True)
	return solve(resultData, False)

def CallRow(row):
	key = 'student' if row['rubric'] else 'rubric'
	r = bootstrapRow(row['dataset'], row['rubric'])
	if r is not None:
		return r[key]
	return None

def bootstrap(dataset, n, rubric=False):
	l = []
	rows = [
		{
			'dataset': dataset,
			'rubric': rubric
		}
	]*n
	bar = Bar('Processing', max=n)
	p = Pool()
	nones = []
	for _, result in enumerate(p.imap_unordered(CallRow, rows)):
		if result is not None:
			l.append(result)
		else:
			nones.append(1)
		bar.next()
	p.close()
	p.join()
	bar.finish()
	return {
		'results': pd.concat(l, ignore_index=True),
		'nones': len(nones)
	}

def getResults(dataset: pd.DataFrame,c=0.025, rubric=False, n=1000):
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
	estimates = solve(dataset)
	if estimates is not None:
		results = bootstrap(dataset, n, rubric)
		r = results['results']
		l = []
		for var in r['Variable'].unique():
			df = r[r['Variable'] == var]['Value']
			ci = (df.quantile(q=c), df.quantile(q=(1-c)))
			transformedci = (logistic(df.quantile(q=c),0), logistic(df.quantile(q=(1-c)),0))
			pvalue = (stats.percentileofscore(df, 0) / 100)*2 if np.mean(df) > 0 else (1 - (stats.percentileofscore(df, 0) / 100))*2
			l.append({
				'Variable': var,
				'Confidence Interval': ci,
				'Logistic Transformed Confidence Interval': transformedci,
				'P-Value': pvalue,
			})
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

def DisplayResults(dataset: pd.DataFrame,c=0.025, rubric=False, n=1000):
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
	rubricR, studentR, bootstrapR, countE, obs, param, AIC, BIC, McFadden, LR, ChiSquared = getResults(dataset, c, rubric, n)
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
	
	x = PrettyTable(align='r')

	x.field_names = ["", "Value",]
	x.add_row(["Number of Observations:", obs])
	x.add_row(["Number of Parameters:", param])
	x.add_row(["AIC:", AIC])
	x.add_row(["BIC:", BIC])
	if McFadden is not None:
		x.add_row(["McFadden R^2:", McFadden])
		x.add_row(["Likelihood Ratio Test Statistic:", LR])
		x.add_row(["Chi-Squared LR P-Value:", ChiSquared])
	else:
		warnings.append('McFadden R^2, Likelihood Ratio Test, and Chi-Squared LR Test could not be displayed because the restricted model could not be solved.')
	print(x)
	if len(warnings) > 0:
		print('Warnings:')
		for warning in warnings:
			print(warning)	
	return (rubricR, studentR, bootstrapR, countE, obs, param, AIC, BIC, McFadden, LR, ChiSquared)

def SaveResults(dataset: pd.DataFrame,c=0.025, rubric=False, n=1000, rubricFile = 'rubric.csv', studentFile = 'student.csv', outputFile = 'output.csv'):
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
	rubricR, studentR, bootstrapR, countE, obs, param, AIC, BIC, McFadden, LR, ChiSquared  = DisplayResults(dataset, c, rubric, n)
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