import nltk
import random
from nltk.corpus import brown, stopwords
import string
from nltk.stem import WordNetLemmatizer
nltk.download('brown')
nltk.download('stopwords')

"""
 Question 4 a) Classification with Naive Bayes
 """
# convert data into list of tuples containing a sentence (list) and a genre (string)
raw_set = []
for genre in brown.categories():
    for fileid in brown.fileids(categories=genre):
        raw_set.append((brown.words(fileids=fileid), genre))

# filter words for stopwords and punctuation, and shuffle
lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')
dataset = [([lemmatizer.lemmatize(word.lower()) for word in text if word not in stopwords
             and word not in string.punctuation and len(lemmatizer.lemmatize(word)) >= 3], category)
           for text, category in raw_set]
all_words = nltk.FreqDist(lemmatizer.lemmatize(word.lower()) for word in brown.words() if word not in stopwords
                          and word not in string.punctuation and len(lemmatizer.lemmatize(word)) >= 3)
random.shuffle(dataset)

# choose desired number of most common words
word_features = list(all_words)[:800]

# print question label after some process so "already up to date" message comes before it
print("Question 4 a) Classification with Naive Bayes:")


def doc_features(doc):
    doc_words = set(doc)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in doc_words)
    return features


# split the dataset
print("Splitting dataset")
split_ratio = 0.7
train_size = int(len(dataset) * split_ratio)
# but first make the dictionary
featureset = [(doc_features(d), c) for (d, c) in dataset]
train_set = featureset[:train_size]
test_set = featureset[train_size:]

# now train and test
print("Training and classifying")
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Testing")
print("Accuracy: ", nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)

"""
 Question 4 b) Classification with Logistic Regression
 """