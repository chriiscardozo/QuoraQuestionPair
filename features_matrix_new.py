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
import numpy as np

def get_sparses_matrices(corpus1, corpus2, corpus3,vocabulary=None):
	vectorizer = CountVectorizer(strip_accents='ascii',ngram_range=(1,3),binary=True,vocabulary=vocabulary)
	vectorizer_equals = CountVectorizer(strip_accents='ascii',ngram_range=(1,1),binary=True,vocabulary=vocabulary)

	print('Building X1...')
	X1 = vectorizer.fit_transform(corpus1)
	corpus1 = 0
	print('Building X2...')
	X2 = vectorizer.fit_transform(corpus2)
	corpus2 = 0
	print('Building X3...')
	X3 = vectorizer_equals.fit_transform(corpus3)
	corpus3 = 0
	return [X1,X2,X3]


def generate_train_features(n_examples):
	print("*** generating matrix train features ***")
	corpus1 = []
	corpus2 = []
	corpus3 = []
	all_words = {}
	st = PorterStemmer()
	y = []
	append_to_X1 = []
	append_to_X2 = []
	append_to_X3 = []

	i = 0

	with open(Constants.TRAIN_TOKENIZED_FILE) as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		for index, line in enumerate(csv_reader):
			q1 = line[3].split(' ')
			q2 = line[4].split(' ')
			y.append(line[5])

			features_equals_count = get_feature_count_equals_words(q1, q2)
			append_to_X1.append([abs(len(q1)-len(q2))/len(q1), features_equals_count[0]])
			append_to_X2.append([abs(len(q1)-len(q2))/len(q2), features_equals_count[1]])
			append_to_X3.append([get_feature_jaccard_distance(q1, q2), get_feature_leveinshtein_distance(q1, q2)])

			for q in [q1, q2]:
				for token in q:
					if token not in all_words:
						all_words[token] = True
			
			equals_words = []
			for t in q1:
				if t in q2:
					equals_words.append(t)

			corpus1.append(' '.join(q1))
			corpus2.append(' '.join(q2))
			corpus3.append(' '.join(equals_words))

			if(i % 1000 == 0): print(str(i) + '\r', end='')
			i += 1
			if(n_examples > 0 and i == n_examples): break

		print('Total distinct words: ', len(all_words))
		X_matrices = get_sparses_matrices(corpus1,corpus2,corpus3)

		# X_matrices[0] = hstack((X_matrices[0],append_to_X1))
		# X_matrices[1] = hstack((X_matrices[1],append_to_X2))
		# X_matrices[2] = hstack((X_matrices[2],append_to_X3))

		for index,item in enumerate(X_matrices):
			print('Dimensions in X' + str(index+1) + ' before SVD: ' + str(item.shape))

			X_aux = item.tocsc()

			X_truncated, Sigma, VT = sparsesvd(X_aux, 30)
			X_matrices[index] = X_truncated.transpose()
			print('New dimensions for X' + str(index+1) + ': ' + str(X_matrices[index].shape))
			# plt.plot(Sigma)
			# plt.show()


		# X_matrices.append(append_to_X1)
		# X_matrices.append(append_to_X2)
		# X_matrices.append(append_to_X3)
		X = np.concatenate(X_matrices,axis=1)

		print('Final X dimentions:', X.shape)

		with open(Constants.TRAIN_MATRIX_FEATURES, 'w') as f:
			csv_writer = csv.writer(f)
			for index, item in enumerate(X):
				line = [index]
				line.extend(item)
				line.append(int(y[index]))
				csv_writer.writerow(line)

def generate_test_features():
	print("*** generating matrix test features ***")