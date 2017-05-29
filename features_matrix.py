import csv
import constants as Constants
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import util as Util
from features_generator import *
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

import numpy, scipy.sparse
from sparsesvd import sparsesvd

def generate_train_features(n_examples):
	print("*** generating matrix train features ***")
	corpus1 = []
	corpus2 = []
	corpus3 = []
	all_words = {}
	st = PorterStemmer()
	y_train = []
	append_to_X = []

	i = 0

	with open(Constants.TRAIN_TOKENIZED_FILE) as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		# next(csv_reader) # jumping the header stuff	e 
		for index, line in enumerate(csv_reader):
			#q1 = tokenize_question(line[3])
			#q2 = tokenize_question(line[4])
			q1 = line[3].split(' ')
			q2 = line[4].split(' ')
			y_train.append(line[5])

			other_X = [abs(len(q1)-len(q2))/len(q1), abs(len(q1)-len(q2))/len(q2), get_feature_jaccard_distance(q1, q2), get_feature_leveinshtein_distance(q1, q2)]
			other_X.extend(get_feature_count_equals_words(q1, q2))
			append_to_X.append(other_X)

			for q in [q1, q2]:
				for token in q:
					if token not in all_words:
						all_words[token] = True
			
			equals_words = []
			for t in q1:
				if t in q2:
					equals_words.append(t)

			corpus3.append(' '.join(equals_words))

			corpus1.append(' '.join(q1))
			corpus2.append(' '.join(q2))
			if(i % 1000 == 0): print(str(i) + '\r', end='')
			i += 1
			if(n_examples > 0 and i == n_examples): break

		vectorizer = CountVectorizer(strip_accents='ascii',ngram_range=(1,3),binary=True)
		vectorizer_equals = CountVectorizer(strip_accents='ascii',ngram_range=(1,1),binary=True)

		print('Total distinct words: ', len(all_words))
		X1 = vectorizer.fit_transform(corpus1)
		X2 = vectorizer.fit_transform(corpus2)
		X3 = vectorizer_equals.fit_transform(corpus3)
		X = hstack([X1,X2,X3])
		print('Dimensions before SVD: ', X.shape)
		
		SVD = TruncatedSVD(n_components=50) 
		X_train = SVD.fit_transform(X)
		Sigma = SVD.explained_variance_ratio_
		VT = SVD.components_
		print(Sigma)

		plt.scatter(range(len(Sigma)), Sigma)
		plt.show()
		exit(0)
		with open(Constants.TRAIN_MATRIX_FEATURES, 'w') as f:
			csv_writer = csv.writer(f)
			for index, item in enumerate(X_train):
				line = [index]
				line.extend(item)
				line.extend(append_to_X[index])
				line.append(int(y_train[index]))
				csv_writer.writerow(line)

def generate_test_features():
	print("*** generating matrix test features ***")
	corpus1 = []
	corpus2 = []
	corpus3 = []
	all_words = {}
	st = PorterStemmer()
	append_to_X = []

	i = 0

	with open(Constants.TEST_TOKENIZED_FILE) as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		# next(csv_reader) # jumping the header stuff	e 
		for index, line in enumerate(csv_reader):
			q1 = line[1].split(' ')
			q2 = line[2].split(' ')

			other_X = [abs(len(q1)-len(q2))/len(q1), abs(len(q1)-len(q2))/len(q2), get_feature_jaccard_distance(q1, q2), get_feature_leveinshtein_distance(q1, q2)]
			other_X.extend(get_feature_count_equals_words(q1, q2))
			append_to_X.append(other_X)

			for q in [q1, q2]:
				for token in q:
					if token not in all_words:
						all_words[token] = True
			
			equals_words = []
			for t in q1:
				if t in q2:
					equals_words.append(t)

			corpus3.append(' '.join(equals_words))

			corpus1.append(' '.join(q1))
			corpus2.append(' '.join(q2))
			if(i % 1000 == 0): print(str(i) + '\r', end='')
			i += 1		

		print('Total distinct words: ', len(all_words))
		X = get_sparse_matrix(corpus1,corpus2,corpus3)
		corpus1 = corpus2 = corpus3 = all_words = None
		print('Dimensions before SVD: ', X.shape)

		# SVD = TruncatedSVD(n_components=10) 
		# X_test = SVD.fit_transform(X)
		# Sigma = SVD.explained_variance_ratio_
		# VT = SVD.components_
		
		X = X.tocsc()
		X_test, Sigma, VT = sparsesvd(X, 10)
		X_test = X_test.transpose()

		# print(Sigma)
		# plt.scatter(range(len(Sigma)), Sigma)
		# plt.show()

		
		with open(Constants.TEST_MATRIX_FEATURES, 'w') as f:
			csv_writer = csv.writer(f)
			for index, item in enumerate(X_test):
				line = [index]
				line.extend(item)
				line.extend(append_to_X[index])
				csv_writer.writerow(line)

def get_sparse_matrix(corpus1, corpus2, corpus3):
	vectorizer = CountVectorizer(strip_accents='ascii',ngram_range=(1,3),binary=True)
	vectorizer_equals = CountVectorizer(strip_accents='ascii',ngram_range=(1,1),binary=True)

	print('Building X1...')
	X1 = vectorizer.fit_transform(corpus1)
	corpus1 = 0
	print('Building X2...')
	X2 = vectorizer.fit_transform(corpus2)
	corpus2 = 0
	print('Building X3...')
	X3 = vectorizer_equals.fit_transform(corpus3)
	corpus3 = 0
	print('Merging X1, X2, X3...')
	return hstack([X1,X2,X3])