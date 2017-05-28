from features_generator import *
import constants as Constants
import csv
import pandas as pd

global_dict = {}
questions_list = []

def generate_train_features(n_examples):
	print("*** generating matrix train featues ***")
	X_train = []
	y_train = []

	i = 0
	with open(Constants.TRAIN_FILE, "r") as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		next(csv_reader) # jumping the header stuff
		for index, line in enumerate(csv_reader):
			q1 = tokenize_question(line[3])
			q2 = tokenize_question(line[4])
			X_train.append(x)
			y_train.append(line[5])
			if(i % 1000 == 0): print(str(i) + '\r', end='')
			i += 1
			if(n_examples > 0 and i == n_examples): break

	print('Dict size: ', len(global_dict))
	exit(0)
	
	X_train = pd.DataFrame(X_train).fillna(0).astype(int).values
	print('Dimentions train: ', len(X_train[0]))

	with open(Constants.TRAIN_MATRIX_FEATURES, 'w') as f:
		csv_writer = csv.writer(f)
		for index, item in enumerate(X_train):
			line = [index] + item.tolist() + [int(y_train[index])]
			csv_writer.writerow(line)

def generate_test_features():
	print("*** generating matrix test featues ***")
	X_test = []

	i = 0
	with open(Constants.TEST_FILE, "r") as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		next(csv_reader) # jumping the header stuff
		for index, line in enumerate(csv_reader):
			x = generate_X(line[3], line[4])
			X_test.append(x)
			if(i % 1000 == 0): print(str(i) + '\r', end='')
			i += 1

	X_test = pd.DataFrame(X_test).fillna(0).astype(int).values
	print('Dimentions test: ', len(X_test[0]))

	with open(Constants.TEST_MATRIX_FEATURES, 'w') as f:
		csv_writer = csv.writer(f)
		for index, item in enumerate(X_test):
			line = [index] + item.tolist()
			csv_writer.writerow(line)