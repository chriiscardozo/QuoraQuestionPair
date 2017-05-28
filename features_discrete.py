from features_generator import *
import constants as Constants
import csv

def generate_X_discrete(q1, q2):
	tokens1 = tokenize_question(q1)
	tokens2 = tokenize_question(q2)

	X = []
	X.append(int(round(10*get_feature_leveinshtein_distance(tokens1, tokens2))))
	X.append(int(round(10*get_feature_jaccard_distance(tokens1, tokens2))))
	counted_words = get_feature_count_equals_words(tokens1, tokens2)
	counted_words[0] = int(counted_words[0]*len(tokens1))
	counted_words[1] = int(counted_words[1]*len(tokens2))
	X += counted_words

	return X

def generate_train_discrete_features(n_examples):
	print("*** generating discrete train featues ***")
	X_train = []
	y_train = []

	i = 0

	with open(Constants.TRAIN_FILE, "r") as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		next(csv_reader) # jumping the header stuff
		for index, line in enumerate(csv_reader):
			x = generate_X_discrete(line[3], line[4])
			X_train.append(x)
			y_train.append(line[5])
			if(i % 1000 == 0): print(str(i) + '\r', end='')
			i += 1
			if(n_examples > 0 and i == n_examples): break

	with open(Constants.TRAIN_DISCRETE_FEATURES, 'w') as f:
		csv_writer = csv.writer(f)
		for index, item in enumerate(X_train):
			line = [index] + item + [int(y_train[index])]
			csv_writer.writerow(line)

def generate_test_discrete_features():
	print("*** generating discrete test features ***")
	X_test = []

	i = 0
	with open(Constants.TEST_FILE, "r") as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		next(csv_reader) # jumping the header stuff
		for index, line in enumerate(csv_reader):
			x = generate_X_discrete(line[1], line[2])
			X_test.append(x)
			if(i % 1000 == 0): print(str(i) + '\r', end='')
			i += 1

	with open(Constants.TEST_DISCRETE_FEATURES, 'w') as f:
		csv_writer = csv.writer(f)
		for index, item in enumerate(X_test):
			line = [index] + item
			csv_writer.writerow(line)