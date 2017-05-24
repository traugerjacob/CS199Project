from pyspark import SparkContext, SparkConf
import sys
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.regression import LinearRegressionWithSGD, RidgeRegressionWithSGD, LassoWithSGD
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.clustering import KMeans
from numpy import array
from math import sqrt
import json
import csv
import numpy
from pyspark.mllib.regression import LabeledPoint
conf = SparkConf().setAppName("Model Selection")
sc = SparkContext(conf = conf)

# JSON Parsing
def jsonCheckParams(row, params):
	for j in params:
		if(row.get(j) == None):
			return False
	return True

def jsonMap(row, params):
	t = ()
	for i in params:
		t = t + (float(row.get(i)),)
	return t

def jsonFilterAndMap(data, params):
	data = data.map(json.loads)
	data = data.filter(lambda x: jsonCheckParams(x, params))
	data = data.map(lambda x: jsonMap(x, params))
	return data


# CSV Parsing
def csvCheckParams(row, params, headerDict):
	for i in params:
		if str(row[headerDict[i]]).isalpha() or row[headerDict[i]] == None:
			return False
	return True

def csvMap(row, params, headerDict):
	t = ()
	for i in params:
		t = t + (float(row[headerDict[i]]),)
	return t

def csvFilterAndMap(data, params):
	data = data.mapPartitions(lambda x: csv.reader(x))
	header = data.first()
	headerDict = {}
	for i in range(len(header)):
		headerDict[header[i]] = i
	data = data.filter(lambda x: csvCheckParams(x, params, headerDict))
	data = data.map(lambda x: csvMap(x, params, headerDict))
	return data

# Returns the Naive Bayes model
def performNaiveBayes(training, test, params):
	model = NaiveBayes.train(training)
	test_preds = (test.map(lambda x: x.label).zip(model.predict(test.map(lambda x: x.features))))
	test_metrics = MulticlassMetrics(test_preds.map(lambda x: (x[0], float(x[1]))))
	testing_accuracy = test_metrics.precision()
	return testing_accuracy



# Returns the Lasso model
def performLasso(training):
	model = LassoWithSGD.train(training, iterations = 100, step = 0.001, intercept = True)
	return model


# Returns the Ridge Regression model
def performRidgeRegression(training):
	model = RidgeRegressionWithSGD.train(training, iterations = 100, step = 0.001, intercept = True)
	return model


# Returns the Linear Regression model
def performLinearRegression(training):
	model = LinearRegressionWithSGD.train(training, iterations = 100, step = 0.001, intercept = True)
	return model


# Returns the best regression model for the dataset given the parameters
def performRegression(data, params):
	training, test = data.randomSplit([.8, .2])
	
	lasso = performLasso(training)
	linReg = performLinearRegression(training)
	ridgeReg = performRidgeRegression(training)
	
	lassoTest = (test.map(lambda x: x.label).zip(lasso.predict(test.map(lambda x: x.features))))
	linTest = (test.map(lambda x: x.label).zip(linReg.predict(test.map(lambda x: x.features))))
	ridgeTest = (test.map(lambda x: x.label).zip(ridgeReg.predict(test.map(lambda x: x.features))))
	
	lassoMetrics = RegressionMetrics(lassoTest.map(lambda x: (x[0], float(x[1]))))
	linMetrics = RegressionMetrics(linTest.map(lambda x: (x[0], float(x[1]))))
	ridgeMetrics = RegressionMetrics(ridgeTest.map(lambda x: (x[0], float(x[1]))))
	
	lassoValue = lassoMetrics.rootMeanSquaredError
	linRegValue = linMetrics.rootMeanSquaredError
	ridgeRegValue = ridgeMetrics.rootMeanSquaredError
	
	if(lassoValue < linRegValue and lassoValue < ridgeRegValue):
		return "lasso"
	if(linRegValue < lassoValue and linRegValue < ridgeRegValue):
		return "linear"
	return "ridge"


# Returns the K-Means model
def performKMeans(data, k):
	kmeans = KMeans.train(data, k)
	return kmeans



# Gets the error of the model
def error(clusters, point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (numpy.asarray(point) - numpy.asarray(center))]))

# Finds the best k-value and its error, if not found, returns k=30 and its error.
#Tries to mimic the elbow method for finding the best k value 
def getKValue(arr):
	diff = .05 * abs(arr[0] - arr[1])
	for i in range(1,len(arr) - 1):
		if(abs(arr[i] - arr[i-1]) <= diff):
			return i+1
	return len(arr)

# Returns the best clustering model for the dataset given the parameters
def performClustering(data, params):
	kmeans_values = []
	# Try k-values from k=1 to k=30
	for k in range(1,31):
		clusters = performKMeans(data, k)
		kmeans_values.append(data.map(lambda point: error(clusters, point)).reduce(lambda x, y: x + y))
		
	bestKMeansK = getKValue(kmeans_values)
	
	return ("KMeans", bestKMeansK)



# MODEL SELECTION ALGORITHM
def modelSelection(argv):
	if len(argv) < 5:
		print("The arguments for this script require:\n" +
				"hdfs:///path/to/filename of the dataset\n" +
				"supervised/unsupervised\n" +
				"classifier/regression/clustering\n" +
				"parameter trying to be guessed\n" +
				"other parameters\n")
	else:
		args = argv[1:]

		#sets up the RDD
		dataset = sc.textFile(args[0])
		params = args[3:]
		if args[0][-3:] == "csv":
			dataset = csvFilterAndMap(dataset, params)
			
		elif args[0][-4:] =="json":
			dataset = jsonFilterAndMap(dataset, params)

		else:
			print("This program only supports .csv and .json files")
			return
		#Model selection algorithm. Currently goes off of scikit learn's cheat sheet
		if args[1] == "supervised":
			labels = dataset.map(lambda x: x[0])
			values = dataset.map(lambda x: x[1:])
			zipped_data = labels.zip(values).map(lambda x: LabeledPoint(x[0], x[1:])).cache()
			datasetTraining, datasetTest = zipped_data.randomSplit([.8, .2])
			
			if args[2] == "classification":
				theModel = NaiveBayes.train(datasetTraining)

				test_preds = (datasetTest.map(lambda x: x.label).zip(theModel.predict(datasetTest.map(lambda x: x.features))))
				predictions = theModel.predict(datasetTest.map(lambda x: x.features))
				test_metrics = MulticlassMetrics(test_preds.map(lambda x: (x[0], float(x[1]))))
				testing_accuracy = test_metrics.precision()
				
				with open('results.txt', 'w+') as f:
					f.write("accuracy: " + str(testing_accuracy) + "\n")
					f.write("confusion matrix:\n" + str(test_metrics.confusionMatrix().toArray()))
				return theModel

			elif args[2] == "regression":
				sample = zipped_data.sample(False, .3)
				model = performRegression(sample, params)
				if(model == "lasso"):
					theModel = LassoWithSGD.train(datasetTraining, iterations = 1000, step = 0.001, intercept = True)
				elif(model == "linear"):
					theModel = LinearRegressionWithSGD.train(datasetTraining, iterations = 1000, step = 0.001, intercept = True)
				else:
					theModel = RidgeRegressionWithSGD.train(datasetTraining, iterations = 1000, step = 0.001, intercept = True)
				test = (datasetTest.map(lambda x: x.label).zip(theModel.predict(datasetTest.map(lambda x: x.features))))
				metrics = RegressionMetrics(test.map(lambda x: (x[0], float(x[1]))))
				value = metrics.rootMeanSquaredError
				with open('results.txt', 'w+') as f:
					f.write(model +" root mean squared error: ")
					f.write(str(value))
				return theModel

			else:
				print("Please use rather classification or regression for supervised learning")
				return

		elif args[1] == "unsupervised":
			if args[2] == "clustering":
				sample = dataset.sample(False, .3)
				model = performClustering(sample, params)
				theModel = performKMeans(dataset, model[-1])
				with open('results.txt', 'w+') as f:
					f.write(str(model))
				return theModel
			else:
				print("Currently this model selection algorithm only supports clustering for unsupervised algorithms")
				return

args = sys.argv
modelSelection(args).save(sc, "./theModel")
