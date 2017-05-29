# -*- coding: utf-8 -*-
import util as Util
import classifiers
import preprocessing
import sys

def main():
	if('tok' in sys.argv):
		start = Util.get_time()
		preprocessing.generate_pre_tokens(train=False,test=False)
		Util.the_time(start, "generating tokens")
	if('pre' in sys.argv):
		start = Util.get_time()
		# pass n_examples=0 to generate all train file
		preprocessing.generate_features(matrix=True,
										n_examples=0,test=False,train=True)
		Util.the_time(start, "preprocessing")

	if('cls' in sys.argv):
		start = Util.get_time()
		classifiers.classify(xgb_learning=True)
		Util.the_time(start, "classify")

if __name__ == "__main__":
	main()
