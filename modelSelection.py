#USE THIS COMMAND TO TEST CSV pyspark modelSelection.py hdfs:///shared/amazon_food_reviews.csv supervised regression Score HelpfulnessNumerator
from pyspark import SparkContext, SparkConf
import sys
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionModel
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.regression import LinearRegressionWithSGD, RidgeRegressionWithSGD, LassoWithSGD, LinearRegressionModel, RidgeRegressionModel, LassoModel
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel
from numpy import array
from math import sqrt
import json
import csv
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
		t = t + (row.get(i),)
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
		t = t + (row[headerDict[i]],)
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


# Returns the Random Forest model
def performRandomForest(training, test, params):
	model = RandomForest.trainClassifier(training, numClasses=2, categoricalFeaturesInfo={},
	                                     numTrees=10, featureSubsetStrategy="auto",
	                                     impurity='gini', maxDepth=4, maxBins=32)
	train_preds = (training.map(lambda x: x.label).zip(model.predict(training.map(lambda x: x.features))))
	test_preds = (test.map(lambda x: x.label).zip(model.predict(test.map(lambda x: x.features))))
	# Create evaluator to compute accuracy
	evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	testing_accuracy = evaluator.evaluate(test_preds)
	return testing_accuracy

def performLogisticRegression(training, test, params):
	model = LogisticRegressionWithSGD.train(training)
	test_preds = test.map(lambda p: (p.label, model.predict(p.features)))
	
	# Create evaluator to compute accuracy
	evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	testing_accuracy = evaluator.evaluate(test_preds)
	return testing_accuracy


# Returns the best model for the data given the parameters
def performClassification(data, params):
	training, test = data.randomSplit([.8, .2])
	naive_bayes = performNaiveBayes(training, test, params)
	#random_forest = performRandomForest(training, test, params)
	# Return the one with the higher testing data accuracy and lower error
	# Need to check for error too because random forest's error is computed independent of accuracy
	#naive_bayes_accuracy = naive_bayes
	#random_forest_accuracy = random_forest
	logistic = performLogisticRegression(training, test, params)
	return "logistic" if logistic > naive_bayes else "Naive Bayes"


# Returns the Lasso model
def performLasso(training):
	model = LassoWithSGD.train(training, iterations = 100, step = 0.001)
	return model


# Returns the Ridge Regression model
def performRidgeRegression(training):
	model = RidgeRegressionWithSGD.train(training, iterations = 100, step = 0.001)
	return model


# Returns the Linear Regression model
def performLinearRegression(training):
	model = LinearRegressionWithSGD.train(training, iterations = 100, step = 0.001)
	return model


# Returns the best regression model for the dataset given the parameters
def performRegression(data, params):
	training, test = data.randomSplit([.8, .2])
	# These should return the error values to test against each other to see which model should be chosen
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
	
	# Returns the regression model
	if(lassoValue < linRegValue and lassoValue < ridgeRegValue):
		return "lasso"
	if(linRegValue < lassoValue and linRegValue < ridgeRegValue):
		return "linear"
	return "ridge"


# Returns the K-Means model
def performKMeans(data, k):
	kmeans = KMeans.train(data, k)
	return kmeans


# Returns the Guassian Mixture model
def performGaussianMixture(data, k):
	gmm = GaussianMixture.train(data, k)
	return gmm


# Gets the error of the model
def error(clusters, point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

# Finds the best k-value and its error, if not found, returns k=30 and its error
def getKValue(arr):
	diff = .1 * abs(arr[0] - arr[1])
	for i in range(1,len(arr) - 1):
		if(abs(arr[i] - arr[i-1]) <= diff):
			return i+1
	return len(arr)

# Returns the best clustering model for the dataset given the parameters
def performClustering(data, params):
	kmeans_values = []
	guassian_mixture_values = []
	# Try k-values from k=1 to k=30
	for k in range(1,31):
		clusters = KMeans.train(data,k)
		kmeans_values.append(data.map(lambda point: error(clusters, point)).reduce(lambda x, y: x + y))
		
		#clusters = GaussianMixture.train(data,k)
		#guassian_mixture_values.append(100000)#data.map(lambda point: error(clusters, point)).reduce(lambda x, y: x + y))
	# Best k-value is calculated when the error difference of two k-values is 10% of the error difference of k=1 and k=2
	# This tries to mimic the elbow method, or where the difference between errors is too small
	bestKMeansK = getKValue(kmeans_values)
	#bestGaussianK, GaussianMixtureError = getKValue(guassian_mixture_values, 0.1 * abs(guassian_mixture_values[1]-guassian_mixture_values[0]))
	# Return the model with the least error
	return ("KMeans", bestKMeansK) #if kMeansError < GaussianMixtureError else ("gaussian", bestGaussianK)


#Added for pyspark testing
def modelSelectionAlternative(path,supervisedOrNot,mlType,parameter,otherParam):
	modelSelection([path,supervisedOrNot,mlType,parameter,otherParam])


# MODEL SELECTION ALGORITHM
def modelSelection(argv):
	if len(argv) < 5:
		print("The arguments for this script require:\n" +
				"path/to/filename of the dataset\n" +
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

		#Model selection algorithm. Currently goes off of scikit learn's cheat sheet
		if args[1] == "supervised":
			labels = dataset.map(lambda x: x[0])
			values = dataset.map(lambda x: x[1:])
			zipped_data = labels.zip(values).map(lambda x: LabeledPoint(x[0], x[1:])).cache()
			datasetTraining, datasetTest = zipped_data.randomSplit([.75, .25])
			sample = zipped_data.sample(False, .3)
			if args[2] == "classification":
				#model = performClassification(sample, params)
				
				#if(model == "Naive Bayes"):
					theModel = NaiveBayes.train(datasetTraining)
					test_preds = (datasetTest.map(lambda x: x.label).zip(theModel.predict(datasetTest.map(lambda x: x.features))))
					test_metrics = MulticlassMetrics(test_preds.map(lambda x: (x[0], float(x[1]))))
					testing_accuracy = test_metrics.precision()
					with open('results.txt', 'w+') as f:
						f.write(str(testing_accuracy))
					return theModel

				#else:

				#	theModel = LogisticRegressionWithSGD.train(datasetTraining)#RandomForest.trainClassifier(datasetTraining, numClasses=2, categoricalFeaturesInfo={},
	                                     #numTrees=10, featureSubsetStrategy="auto",
	                                     #impurity='gini', maxDepth=4, maxBins=32)
			elif args[2] == "regression":
				model = performRegression(sample, params)
				if(model == "lasso"):
					theModel = LassoWithSGD.train(datasetTraining, iterations = 1000, step = 0.001)
				elif(model == "linear"):
					theModel = LinearRegressionWithSGD.train(datasetTraining, iterations = 1000, step = 0.001)
				else:
					theModel = RidgeRegressionWithSGD.train(datasetTraining, iterations = 1000, step = 0.001)
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
			sample = dataset.sample(False, .3)
			if args[2] == "clustering":
				model = performClustering(sample, params)
				
				if(model[0] == "gaussian"):
					theModel = GuassianMixture.train(dataset, model[1])
				else:
					theModel = KMeans.train(dataset, model[1])
				with open('results.txt', 'w+') as f:
					f.write(str(model))
				return theModel
			else:
				print("Currently this model selection algorithm only supports clustering for unsupervised algorithms")
				return

modelSelection(sys.argv)
