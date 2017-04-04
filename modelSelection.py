#from pyspark import SparkContext, SparkConf
import sys
#conf = SparkConf().setAppName("Model Selection")
#sc = SparkContext(conf = conf)

if len(sys.argv) < 5:
	print("The arguements for this script require(if there is a space in the name of a parameter please replace the space with a +):\nfilename of the dataset\nsupervised/unsupervised\nclassifier/regression\nparameter trying to be guessed\nother parameters")
else:
	args = sys.argv[1:]
	
