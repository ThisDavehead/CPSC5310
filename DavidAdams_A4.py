
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

import nltk
import random
from nltk.corpus import brown, stopwords
import string
from nltk.stem import WordNetLemmatizer

import gensim
import gensim.downloader as api
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import pandas as pd
import numpy as np
import re

nltk.download('brown')
nltk.download('stopwords')

# convert data into list of tuples containing a sentence (list) and a genre (string)
lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')
dataset = []
for genre in brown.categories():
    for sent in brown.sents(categories=genre):
        sentstr = ' '.join([lemmatizer.lemmatize(word.lower()) for word in sent if word not in stopwords
                            and word not in string.punctuation and
                            len(lemmatizer.lemmatize(word)) >= 3])
        dataset.append((sentstr, genre))

random.shuffle(dataset)

print("3a) Logistic regression")

# convert dataset list to dataframe
ds = pd.DataFrame(list(dataset), columns=["sentence", "genre"])

# define input and output for model training
sentences = ds['sentence'].values
genres = ds['genre'].values

# tag sentences
labeled_sentences = []
for index, datapoint in ds.iterrows():
    tokenized_words = re.findall(re.compile(r"\w+", re.I), datapoint["sentence"].lower())
    labeled_sentences.append(TaggedDocument(words=tokenized_words, tags=['SENT_%s' % index]))

# train Doc2Vec model on tagged sentences
model = Doc2Vec(min_count=1, window=10, vector_size=100, sample=1e-4, negative=5, workers=8, epochs=10)
model.build_vocab(labeled_sentences)
print("training Word2Vec model...")
model.train(labeled_sentences, total_examples=model.corpus_count, epochs=10)
print("training complete")

# vectorize sentences using doc2vec
y = []
vectorized_sentences = []
for i in range(0, ds.shape[0]):
    label = 'SENT_%s' % i
    vectorized_sentences.append(model.docvecs[label])
ds['vectorized_comments'] = vectorized_sentences
X_full = ds["vectorized_comments"].T.to_list()
Y_full = ds['genre'].values

# encode genres for classification
encoder = LabelEncoder()
encoder.fit(Y_full)
Y_full = encoder.transform(Y_full)
X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.1)

# split the training and test data using sklearn utility
X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.1)

# declare and train the Logistic Regression model we're using
classifier = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)
classifier.fit(X_train, Y_train)
score = classifier.score(X_test, Y_test)
np.set_printoptions(precision=2)
print("Logistic Regression accuracy:", score)
print(" ")

# ----------------------------------------------------------------------------------
print("3b) Multi-layer neural network")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(3, 3, 2), random_state=1)

clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)
np.set_printoptions(precision=2)
print("Multi-Layer Neural Network accuracy:", score)
