from pyspark import SparkContext, SparkConf
import sys
conf = SparkConf().setAppName("Model Selection")
sc = SparkContext(conf = conf)

def main(argv):
		#This is a lot of parameters needed for the model selection process. If there are any that can be removed talk to partner about it to see if it can be done
	if len(argv) < 5:
		print("The arguements for this script require(if there is a space in the name of a parameter please replace the space with a +):\npath/to/filename of the dataset\nsupervised/unsupervised\nclassifier/regression\nparameter trying to be guessed\nother parameters")
	else:
		args = argv[1:]
		dataset = sc.textFile(args[0])
		if(args[2] == "classification"):
			#code goes here to find model selection for classification
			print(-1)
			return -1 #delete this when code is written
		if(args[2] == "regression"):
			#code goes here to find model selection for regression
			print(-1)
			return -1 #delete this when code is written
		else:
			print("Please use rather classification or regression")
			return;




main(sys.argv)
