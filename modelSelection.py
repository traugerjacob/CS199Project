from pyspark import SparkContext, SparkConf
import sys
conf = SparkConf().setAppName("Model Selection")
sc = SparkContext(conf = conf)

def main(argv):
		#This is a lot of parameters needed for the model selection process. If there are any that can be removed talk to partner about it to see if it can be done
	if len(argv) < 5:
#idk if there is an unsupervised regression algorithm so if there is then add itand change the code around to accomidate it
		print("The arguements for this script require(if there is a space in the name of a parameter please replace the space with a +):\npath/to/filename of the dataset\nsupervised/unsupervised\nclassifier/regression/clustering\nparameter trying to be guessed\nother parameters")
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
		
		#Model selection algorithm. Currently goes off of scikit learn's cheat sheet
		if(args[1] == "supervised"):
			if(args[2] == "classification"):
				if(count(dataset) > 100000):
					text = ""
					while(text !="y" or text !="n"):
						text = input("Text data? y/n")
					if(text = "y")
						return "Naive Bayes"
					else:
						return "KNeighbors Classifier and if that doesnt work then SVC or Ensemble Classifiers"

			if(args[2] == "regression"):
				if(count(dataset) > 100000):
					if (len(args) < 6):
						return "Lasso"
					else:
						return "Ridge Regression/SVR and then Ensemble regressors"
			else:
				print("Please use rather classification or regression")
				return;
		if(args[1] == "unsupervised"):
			if(args[2]) == "clustering":
				text = ""
				while(text !="y" or text !="n"):
					text = input("Text data? y/n")
				if(text = "y")
					return "Latent Dirichlet Allocation"
				else:
					return "KMeans and if that does not work than Guassian Mixture Modeling"
			else:
				print("please use clusetering")




main(sys.argv)
