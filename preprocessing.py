import util as Util
import features_continuous as Continuous
import features_discrete as Discrete
import features_matrix as Matrix

def generate_features(continuous=False,discrete=False,sparse_matrix=False,train=True,test=True,n_examples=10000):
	if(continuous):
		if(train):
			start = Util.get_time()
			Continuous.generate_train_continuous_features(n_examples)
			Util.the_time(start, "generating continuous train features")
		if(test):
			start = Util.get_time()
			Continuous.generate_test_continuous_features()
			Util.the_time(start, "generating continuous test features")
	if(discrete):
		if(train):
			start = Util.get_time()
			Discrete.generate_train_discrete_features(n_examples)
			Util.the_time(start, "generating discrete train features")
		if(test):
			start = Util.get_time()
			Util.the_time(start, "generating discrete test features")
	if(sparse_matrix):
		if(train):
			start = Util.get_time()
			Matrix.generate_train_features(n_examples)
			Util.the_time(start, "generating matrix train features")
		if(test):
			start = Util.get_time()
			Matrix.generate_test_features()
			Util.the_time(start, "generating matrix test features")