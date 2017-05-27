import util as Util
import classifiers
import preprocessing
import sys

def main():
	if('pre' in sys.argv):
		start = Util.get_time()
		preprocessing.generate_features(n_examples=10000,test=False) # pass n_examples=0 to generate all train file
		Util.the_time(start, "preprocessing")

	if('cls' in sys.argv):
		start = Util.get_time()
		classifiers.classify(neural_learning=True)
		Util.the_time(start, "classify")

if __name__ == "__main__":
	main()
