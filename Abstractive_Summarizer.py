import cPickle as pickle #data preprocessing
from collections import Counter #tokenization
import keras #ML
import postprocessing as pr #helper

#step 1 - Load data
with open('data/%s.pkl', 'rb') as fp:
	heads, desc, keywords = pickle.load(fp)

#headings
i=0
heads[1]

#Source
desc[i]

#Tokenize text
def get_vocab(1st)
	vocabcount, vocab = Counter(w for txt in lst for w in txt.split())
	return vocab, vocabcount

vocab, vocabcount = get_vocab(heads+desc)

print vocab[:5]
print '...', len(vocab)

#Create word embeddings with GloVe
path = 'glove.6B.zip'
glove_weights = get_glove_weights(path, orirgin='http link to glove')
word_embeddings = pr.build_glove_matrix(glove_weights, vocab)

#3 stacked LSTM RNN (recurrent neural network)
def build_model(embedding):
model = Sequential()
model.add(Embedding(weights=[embedding], name='embedding_1'))
for i in range(3):
	lstm = LSTM(rnn_size,
				name = 'lstm_%d'%(i+1))
			model.add(lstm)
			model.add(Dropout(p_dense,name='dropout_%d'%(i+1)))
		model.add(Dense())
		model.add(Activation('softmax', name='activation'))
		return model

#Initialize Encoder RNN with Embeddings
encoder = build_model(word_embeddings)
encoder.compile(loss='categorical_crossentropy', optimizer='rmsprop')
encoder.save_weights('embeddings.pkl', overwrite=True)


#Initialize Decoder RNN with Embeddings
with open('embeddings.pkl', 'rb')
	embeddings = pickle.load(fp)
decoder = build_model(embeddings)

#Conver a given article to a headline
headline1 = pr.gen_headlin(decoder, desc[1])

#Convert a given article to a headline
headline2 = pr.gen_headline(decoder, desc[2])






