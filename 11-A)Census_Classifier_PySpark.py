##/ fitting census classifier model in Pyspark 

censusRdd = sc.textFile('census.data')
censusRddSplit = censusRdd.map(lambda x: [e.strip() for e in x.split(',')])
categoricalFeatures = [e for e,i in enumerate(censusRddSplit.take(1)[0]) if i.isdigit()==False]
allFeatures = [e for e,i in enumerate(censusRddSplit.take(1)[0])]
categoricalMaps = []
for c in categoricalFeatures:
	catDict = censusRddSplit.map(lambda x: x[c] if len(x) > c else None).\
	filter(lambda x: x is not None).\
	distinct().\
	zipWithIndex().\
	collectAsMap()
	censusRddSplit.map(lambda x: x[c]).take(1)
	categoricalMaps.append(catDict)
	
expandedFeatures = 0
for c in categoricalMaps:
	expandedFeatures += len(c)
	expandedFeatures += len(allFeatures)-len(categoricalFeatures)-2


def formatPoint(p):
	if p[-1] == '<=50K':
		label = 0
	else:
		label = 1
		vector = [0.0]*expandedFeatures
		categoricalIndex = 0
		categoricalVariable = 0
		for e,c in enumerate(p[:-1]):
			if e in categoricalFeatures:
				vector[categoricalIndex + categoricalMaps[categoricalVariable][c]]=1
				categoricalIndex += len(categoricalMaps[categoricalVariable])
				categoricalVariable +=1
			else:
				vector[e] = c
				categoricalIndex += 1
				return LabeledPoint(label,vector)

censusRddLabeled = censusRddSplit.map(lambda x: formatPoint(x))

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
censusLogistic = LogisticRegressionWithLBFGS.train(censusRddLabeled )
censusLogistic.weights
