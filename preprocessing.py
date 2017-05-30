import util as Util
import features_continuous as Continuous
import features_discrete as Discrete
import features_matrix_new as Matrix

import csv
import constants as Constants
from features_generator import tokenize_question

def generate_features(continuous=False,discrete=False,matrix=False,train=True,test=True,n_examples=10000):
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
	if(matrix):
		if(train):
			start = Util.get_time()
			Matrix.generate_train_features(n_examples)
			Util.the_time(start, "matrix train features")
		if(test):
			start = Util.get_time()
			Matrix.generate_test_features()
			Util.the_time(start, "matrix test features")

def generate_pre_tokens(train=False,test=False):
	if(train): generate_pre_tokens_train()
	if(test): generate_pre_tokens_test()

def generate_pre_tokens_test():
	i = 0
	f_out = open(Constants.TEST_TOKENIZED_FILE, 'w')
	with open(Constants.TEST_FILE, 'r') as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		next(csv_reader) # jumping the header stuff	e 
		for index, line in enumerate(csv_reader):
			q1 = tokenize_question(line[1])
			q2 = tokenize_question(line[2])

			q1 = ' '.join(q1)
			q2 = ' '.join(q2)

			row = []
			row.append(int(line[0]))
			row.extend([q1, q2])

			csv_writer = csv.writer(f_out, quotechar='"', quoting=csv.QUOTE_NONNUMERIC,delimiter=',')
			csv_writer.writerow(row)
			i += 1
			if(i % 1000 == 0): print(str(i) + '\r', end='')


def generate_pre_tokens_train():
	i = 0
	f_out = open(Constants.TEST_TOKENIZED_FILE, 'w')
	with open(Constants.TRAIN_FILE, 'r') as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		ext(csv_reader) # jumping the header stuff	e 
		for index, line in enumerate(csv_reader):
			q1 = tokenize_question(line[3])
			q2 = tokenize_question(line[4])

			q1 = ' '.join(q1)
			q2 = ' '.join(q2)

			row = []
			row.extend([int(x) for x in line[:3]])
			row.extend([q1, q2, int(line[5])])

			csv_writer = csv.writer(f_out, quotechar='"', quoting=csv.QUOTE_NONNUMERIC,delimiter=',')
			csv_writer.writerow(row)
			i += 1
			if(i % 1000 == 0): print(str(i) + '\r', end='')