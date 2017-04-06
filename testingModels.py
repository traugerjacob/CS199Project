from pyspark import SparkContext, SparkConf
import sys
conf = SparkConf().setAppName("Testing Models")
sc = SparkContext(conf = conf)

"""
Use this file to 'prove' that our Model Selection works as specified.
"""
