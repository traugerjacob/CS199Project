#MIGHT BE FIXED HAS NOT BEEN TESTED TODO find a way to filter the dataset to only get the parameters we want left in them
#TODO split the dataset into training and testing data (also when finding the best model should we train on a smaller dataset to pick the best model and then use that model on the full dataset?)
#TODO fill in the stubbed functions for the models

from pyspark import SparkContext, SparkConf
import sys
conf = SparkConf().setAppName("Model Selection")
sc = SparkContext(conf = conf)



def filtered(data, params):
	data = data.filter(lambda x: checkParams(x, params))
	return data

def jsonCheckParams(row, params):
	for j in params:
		if(row.get(j) == None):
			return False
	return True

def jsonMap(row, params)
	t = ()
	for i in params:
		t = t + (row.get(i),)
	return t

def jsonFilterAndMap(data, params):
	data = data.filter(lambda x: jsonCheckParams(x, params))
	data = data.map(lambda x: jsonMap(x, params))
	return data



def csvCheckParams(row, params):
	for i in row:
		if i == None or i in params:
			return False
	return True

def csvMap(row, params, headerDict):
	t = ()
	for i in params:
		t = t + (row[headerDict[params]],)
	return t

def csvFilterAndMap(data, params):
	header = data.first().collect()
	data = data.filter(lambda x: csvCheckParams(x, params))
	headerDict = {}
	for i in range(len(header[0])):
		headerDict[header[0][i]] = i
	data = data.map(lambda x: csvMap(x, params, headerDict))

#returns the Naive Bayes model
def performNaiveBayes(data, params):
	return None


#returns the Random Forest model
def performRandomForest(data, params):
	return None


#returns the best model for the data given the parameters
def performClassification(data, params):
	from pyspark.mllib.regression import LabeledPoint
	from pyspark.mllib.classification import NaiveBayes
	from pyspark.mllib.evaluation import MulticlassMetrics

	naiveBayes = performNaiveBayes(data, params)
	randomForest = performRandomForest(data, params)
	return None





#returns the Lasso model
def performLasso(data, params):
	return None


#returns the Ridge Regression model
def performRidgeRegression(data, params):
	return None


#returns the Linear Regression model
def performLinearRegression(data, params):
	return None


#returns the best regression model for the dataset given the parameters
def performRegression(data, params):
	lasso = performLasso(dataset, params)
	linReg = performLinearRegression(dataset, params)
	ridgeReg = perfromRidgeRegression(dataset, params)
	return None





#returns the K-Means model
def performKMeans(data, k):
	kMeans = KMeans.train(data, k)


#reutrns the Guassian Mixture model
def performGaussianMixture(data, k):
	gmm = GaussianMixture.train(data, k)
	return gmm

#returns the best clustering model for the dataset given the parameters
def performClustering(data, params):
	from pyspark.mllib.clustering import KMeans, KMeansModel
	from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel
	
	#Make sure to find the best k value.
	bestKMeans = 2
	bestKGaussian = 2
	kMeansBest = -1
	guassianBest = -1
	for k in range(2,20):
		kMeans = performKMeans(data, k)
		guassian = performGuassianMixture(data, k)
		#TODO do some computations to see if this k is best (it isnt going to be the best value but where right before where the dimishing returns is too great)
		#And see if KMeans or guassian does better with each of their respective best k values
	if guassian < kMeans:
		return ("guassian", bestKGuassian)
	else:
		return ("kMeans", bestKMeans)
	


def main(argv):
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
		params = argv[3:]
		if args[-3:] == "csv":
			import csv
			dataset = dataset.mapPartitions(lambda x: csv.reader(x))
			dataset = csvFilterAndMap(dataset, params)

		elif args[-4:] =="json":
			import json
			dataset = dataset.map(json.loads)
			dataset = jsonFilterAndMap(dataset, params)
		
		
		
		#Model selection algorithm. Currently goes off of scikit learn's cheat sheet
		if args[1] == "supervised":
			if args[2] == "classification":
				model = performClassification(dataset, params)
				#TODO predict the model across the entire dataset

			if args[2] == "regression":
				model = performRegression(dataset, params)		
				#TODO predict the model across the entire dataset
			
			else:
				print("Please use rather classification or regression for supervised learning")
				return

		if args[1] == "unsupervised":
			if args[2] == "clustering":
				model = perfromClustering(dataset, params)
				if(model[0] == "gaussian"):
					theModel = GuassianMixture.train(datasetTraining, model[1])
				else:
					theModel = KMeans.train(datasetTraining, model[1])

				return theModel
			else:
				print("Currently this model selection algorithm only supports clustering for unsupervised algorithms")
				return


main(sys.argv)
