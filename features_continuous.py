from features_generator import *
import constants as Constants
import csv

def generate_X_continuous(q1, q2):
	tokens1 = tokenize_question(q1)
	tokens2 = tokenize_question(q2)

	X = []
	X.append(get_feature_leveinshtein_distance(tokens1, tokens2))
	X.append(get_feature_jaccard_distance(tokens1, tokens2))
	X += get_feature_count_equals_words(tokens1, tokens2)

	return X

def generate_train_continuous_features(n_examples):
	print("*** generating continuous train featues ***")
	X_train = []
	y_train = []

	i = 0

	with open(Constants.TRAIN_FILE, "r") as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		next(csv_reader) # jumping the header stuff
		for index, line in enumerate(csv_reader):
			x = generate_X_continuous(line[3], line[4])
			X_train.append(x)
			y_train.append(line[5])
			if(i % 1000 == 0): print(str(i) + '\r', end='')
			i += 1
			if(n_examples > 0 and i == n_examples): break

	with open(Constants.TRAIN_CONTINUOUS_FEATURES, 'w') as f:
		csv_writer = csv.writer(f)
		for index, item in enumerate(X_train):
			line = [index] + item + [int(y_train[index])]
			csv_writer.writerow(line)

def generate_test_continuous_features():
	print("*** generating continuous test features ***")
	X_test = []

	i = 0
	with open(Constants.TEST_FILE, "r") as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		next(csv_reader) # jumping the header stuff
		for index, line in enumerate(csv_reader):
			x = generate_X_continuous(line[1], line[2])
			X_test.append(x)
			if(i % 1000 == 0): print(str(i) + '\r', end='')
			i += 1

	with open(Constants.TEST_CONTINUOUS_FEATURES, 'w') as f:
		csv_writer = csv.writer(f)
		for index, item in enumerate(X_test):
			line = [index] + item
			csv_writer.writerow(line)