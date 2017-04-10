from pyspark import SparkContext, SparkConf
import sys
conf = SparkConf().setAppName("Model Selection")
sc = SparkContext(conf = conf)


def checkParams(row):
	for i in row:
		if i == None:
			return False
	return True

def filtered(data, params):
	data = data.filter(lambda x: checkParams(x))
	return data




#returns the Naive Bayes model
def performNaiveBayes(data, params):
	return None


#returns the Random Forest model
def performRandomForest(data, params):
	return None


#returns the best model for the data given the parameters
def performClassification(data, params):
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
def performKMeans(data, params):
	return None


#reutrns the Guassian Mixture model
def performGaussianMixture(data, params):
	return None

#returns the best clustering model for the dataset given the parameters
def performClustering(data, params):
	kMeans = performKMeans(data, params)
	guassian = performGuassianMixture(data, params)
	return None


def main(argv):
		#This is a lot of parameters needed for the model selection process. If there are any that can be removed talk to partner about it to see if it can be done
	if len(argv) < 5:
		#idk if there is an unsupervised regression algorithm so if there is then add itand change the code around to accomidate it
		print("The arguements for this script require(replace sapce in parameters with a '+'):\n" +
				"path/to/filename of the dataset\n" +
				"supervised/unsupervised\n" +
				"classifier/regression/clustering\n" +
				"parameter trying to be guessed\n" +
				"other parameters\n")
	else:
		args = argv[1:]

		#sets up the RDD
		dataset = sc.textFile(args[0])
		if(args[-3:] == "csv"):
			import csv
			dataset = dataset.mapPartitions(lambda x: csv.reader(x))

		elif(args[-4:] =="json"):
			import json
			dataset = dataset.map(json.loads)

		params = argv[3:]
		
		#filters dataset to get all None values out
		dataset = filtered(dataset, params)
		
		#Model selection algorithm. Currently goes off of scikit learn's cheat sheet
		if(args[1] == "supervised"):
			if(args[2] == "classification"):
				model = performClassification(dataset, params)
				#TODO predict the model across the entire dataset
			if(args[2] == "regression"):
				model = performRegression(dataset, params)		
				#TODO predict the model across the entire dataset
			else:
				print("Please use rather classification or regression for supervised learning")
				return

		if(args[1] == "unsupervised"):
			if(args[2]) == "clustering":
				model = perfromClustering(dataset, params)		
				#TODO predict the model across the entire dataset
			else:
				print("Currently this model selection algorithm only supports clustering for unsupervised algorithms")
				return
main(sys.argv)
