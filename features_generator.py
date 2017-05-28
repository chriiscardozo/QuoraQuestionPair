import distance
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import util as Util

def get_feature_jaccard_distance(tokens1, tokens2):
	return 0 if(len(tokens1) == 0 and len(tokens2) == 0) else distance.jaccard(tokens1, tokens2)

def get_feature_leveinshtein_distance(tokens1, tokens2):
	return distance.nlevenshtein(tokens1, tokens2)

def get_feature_count_equals_words(tokens1, tokens2):
	count = 0
	for item in tokens1:
		if item in tokens2:
			count += 1
	return [0 if len(tokens1) == 0 else count/len(tokens1),
			0 if len(tokens2) == 0 else count/len(tokens2)]

def tokenize_question(q):
	sentences = sent_tokenize(q)
	tokens = []
	for s in sentences:
		tokenizer = RegexpTokenizer(r'\w+')
		tokens += tokenizer.tokenize(s.lower())
	tokens_filtred =[x for x in tokens if x not in Util.ENGLISH_STOP_WORDS]

	st = PorterStemmer()
	return [st.stem(x) for x in tokens_filtred]