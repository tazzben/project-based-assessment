import pandas as pd
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool
from progress.bar import Bar

def itemPb(q, s, k, b):
	return (1 - q - s)^(np.floor(k)) * (q + s + (q + s - 1) * np.ceil(-k/b))

def opFunction(x, data):
	return -1.0 * (np.array([ itemPb(x[item[1]], x[item[2]], item[0], item[3]) for item in data ]).prod())

def solve(dataset):
	studentCode, uniqueStudents = pd.factorize(dataset['student'])
	questionCode, uniqueQuestion = pd.factorize(dataset['rubric'])
	questionCode = questionCode + uniqueStudents.size
	map = np.concatenate((uniqueStudents, uniqueQuestion), axis=None).tolist()
	data = list(zip(dataset['k'], studentCode, questionCode, dataset['bound']))
	minValue = minimize(opFunction, [0]*len(map), args=(data, ), method='Powell')
	if (minValue.success):
		fullResults = list(zip(map, minValue.x))
		studentResults = fullResults[:uniqueStudents.size]
		questionResults = fullResults[uniqueStudents.size:]
		return {
			'student': pd.DataFrame(studentResults, columns=['Variable', 'Value']),
			'rubric': pd.DataFrame(questionResults, columns=['Variable', 'Value']),
			'AIC': 2*(uniqueStudents.size+uniqueQuestion.size)-2*np.log(minValue.fun),
			'BIC': (uniqueStudents.size+uniqueQuestion.size)*np.log(len(studentCode))-2*np.log(minValue.fun),
			'n': len(studentCode),
			'NumberOfParameters': (uniqueStudents.size+uniqueQuestion.size)
		}
	else:
		return None

def bootstrapRow (dataset, rubric=False):
	key = 'rubric' if rubric else 'student'
	ids = dataset[key].unique()
	randomGroupIds = np.random.choice(ids, size=len(ids), replace=True)
	l = []
	for c, i in enumerate(randomGroupIds):
		rows = dataset[dataset[key]==i]
		rows[key] = c
		l.append(rows)
	resultData = pd.concat(l, ignore_index=True)
	return solve(resultData)

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

def getResults(dataset,c=0.025, rubric=False, n=10000):
	dataset.dropna(inplace=True)
	if set(['k','bound', 'student', 'rubric']).issubset(df.columns) is False:
		raise Exception('Invalid pandas dataset, missing columns. k, bound, student, and rubric are required.')
	dataset = dataset[dataset.k.apply(lambda x: x.isnumeric())]
	dataset = dataset[dataset.bound.apply(lambda x: x.isnumeric())]
	estimates = solve(dataset)
	if estimates is not None:
		r = bootstrap(dataset, n, rubric)
		l = []
		for var in r['Variable'].unique():
			df = r[r['Variable'] == var]['Value']
			ci = (df.quantile(q=c), df.quantile(q=(1-c)))
			l.append({
				'Variable': var,
				'Confidence Interval': ci,
			})
		return (estimates['rubric'], estimates['student'], pd.DataFrame(l), r['nones'], estimates['n'], estimates['NumberOfParameters'], estimates['AIC'], estimates['BIC'])
	else:
		raise Exception('Could not find estimates.')

def DisplayResults(dataset,c=0.025, rubric=False, n=10000):
	rubricR, studentR, bootstrapR, countE, obs, param, AIC, BIC  = getResults(dataset, c, rubric, n)
	print('Rubric Estimates:')
	print(rubricR)
	print('Student Estimates:')
	print(studentR)
	print('Bootstrap Results:')
	print(bootstrapR)
	if countE > 0:
		print('Warning: ' + str(countE) + ' of ' + str(n) + ' bootstrap samples were empty.')
	print('Number of Observations:', str(obs))
	print('Number of Parameters:', str(param))
	print('AIC:', str(AIC))
	print('BIC:', str(BIC))
