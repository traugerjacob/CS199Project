#MIGHT BE FIXED HAS NOT BEEN TESTED TODO find a way to filter the dataset to only get the parameters we want left in them
#TODO split the dataset into training and testing data (also when finding the best model should we train on a smaller dataset to pick the best model and then use that model on the full dataset?)
#TODO fill in the stubbed functions for the models

from pyspark import SparkContext, SparkConf
import sys
conf = SparkConf().setAppName("Model Selection")
sc = SparkContext(conf = conf)


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
	data = data.map(json.loads)
	data = data.filter(lambda x: jsonCheckParams(x, params))
	data = data.map(lambda x: jsonMap(x, params))
	return data



def csvCheckParams(row, params, headerDict):
	for i in params:
		if row[headerDict[params]] == None
			return False
	return True

def csvMap(row, params, headerDict):
	t = ()
	for i in params:
		t = t + (row[headerDict[params]],)
	return t

def csvFilterAndMap(data, params):
	data = data.mapPartitions(lambda x: csv.reader(x))
	header = data.first()
	data = data.subtract(header)
	header = header.collect()
	headerDict = {}
	for i in range(len(header[0])):
		headerDict[header[0][i]] = i
	data = data.filter(lambda x: csvCheckParams(x, params, headerDict))
	data = data.map(lambda x: csvMap(x, params, headerDict))

#returns the Naive Bayes model
def performNaiveBayes(training, test, params):
	model = NaiveBayes.train(training)

	train_preds = (training.map(lambda x: x.label).zip(model.predict(training.map(lambda x: x.features))))
	test_preds = (test.map(lambda x: x.label).zip(model.predict(test.map(lambda x: x.features))))
	trained_metrics = MulticlassMetrics(train_preds.map(lambda x: (x[0], float(x[1]))))
	test_metrics = MulticlassMetrics(test_preds.map(lambda x: (x[0], float(x[1]))))

	# Gets the accuracy of the data (first metric?)
	output = str(trained_metrics.confusionMatrix().toArray()) + '\n' +
			 'Training precision: ' + str(trained_metrics.precision()) + '\n' +
			 str(test_metrics.confusionMatrix().toArray()) + '\n' +
			 'Testing precision: ' + str(test_metrics.precision()) + '\n'

	return output

#returns the Random Forest model
def performRandomForest(data, params):
	return None


#returns the best model for the data given the parameters
def performClassification(data, params):
	from pyspark.mllib.classification import NaiveBayes
	from pyspark.mllib.tree import RandomForest, RandomForestModel
	from pyspark.mllib.evaluation import MulticlassMetrics
	
	training, test = data.randomSplit([.8, .2])

	naiveBayes = performNaiveBayes(training, test, params)
	randomForest = performRandomForest(training, test, params)
	#TODO find out which is the best model and return it




#returns the Lasso model
def performLasso(training, test):
	model = LassoWithSGD.train(training, iterations = 100, step = 0.00000001)
	return model


#returns the Ridge Regression model
def performRidgeRegression(training, test):
	model = RidgeRegressionWithSGD.train(data, iterations = 100, step = 0.00000001)
	return model

#returns the Linear Regression model
def performLinearRegression(training, test):
	model = LinearRegressionWithSGD.train(data, iterations = 100, step = 0.00000001)
	return model


#returns the best regression model for the dataset given the parameters
def performRegression(data, params):
	from pyspark.mllib.regression import LinearRegressionWithSGD, RidgeRegressionWithSGD, LassoWithSGD
	
	training, test = data.randomSplit([.8, .2])

	#These should return the error values to test against each other to see which model should be chosen
	lasso = performLasso(training, test, params)
	linReg = performLinearRegression(training, test, params)
	ridgeReg = perfromRidgeRegression(training, test, params)
	
	#TODO find out which did the best and return it
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
			dataset = csvFilterAndMap(dataset, params)

		elif args[-4:] =="json":
			import json
			dataset = jsonFilterAndMap(dataset, params)



		#Model selection algorithm. Currently goes off of scikit learn's cheat sheet
		if args[1] == "supervised":
			from pyspark.mllib.regression import LabeledPoint
			
			labels = data.map(lambda x: x[0])
			values = data.map(lambda x: x[1:])
			zipped_data = labels.zip(values).map(lambda x: LabeledPoint(x[0], x[1:])).cache()

			datasetTraining, datasetTest = zipped_data.randomSplit([.75, .25])
			
			sample = zipped_data.sample(False, .3)
			

			if args[2] == "classification":
				model = performClassification(sample, params)
				
				if(model == "Naive Bayes"):
					theModel = NaiveBayes.train(training)
	
				else:
					#TODO implement randomForest

			if args[2] == "regression":
				model = performRegression(sample, params)
				#TODO predict the model across the entire dataset
				if(model == "lasso"):
					theModel = LassoWithSGD.train(training, iterations = 100, step = 0.00000001)
				
				elif(model == "linear"):
					theModel = LinearRegressionWithSGD.train(data, iterations = 100, step = 0.00000001)
				
				else:
					theModel = RidgeRegressionWithSGD.train(data, iterations = 100, step = 0.00000001)



			else:
				print("Please use rather classification or regression for supervised learning")
				return

		if args[1] == "unsupervised":
			if args[2] == "clustering":
				model = perfromClustering(sample, params)
				if(model[0] == "gaussian"):
					theModel = GuassianMixture.train(datasetTraining, model[1])
				else:
					theModel = KMeans.train(datasetTraining, model[1])

				return theModel
			else:
				print("Currently this model selection algorithm only supports clustering for unsupervised algorithms")
				return


main(sys.argv)
