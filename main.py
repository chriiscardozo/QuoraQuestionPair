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
		preprocessing.generate_features(matrix=True,n_examples=0,test=False,train=True)
		Util.the_time(start, "preprocessing")

	if('cls' in sys.argv):
		start = Util.get_time()
		nb = xgb = nn = sub = False
		if('nb' in sys.argv): nb = True
		if('xgb' in sys.argv): xgb = True
		if('nn' in sys.argv): nn = True
		if('sub' in sys.argv): sub = True
		classifiers.classify(nb_learning=nb,xgb_learning=xgb,nn_learning=nn,submission=sub)
		Util.the_time(start, "classify")

if __name__ == "__main__":
	main()
