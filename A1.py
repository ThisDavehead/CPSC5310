import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('brown')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
stemmer = PorterStemmer()


""" Word Tokens per Genre """
print("------------------")
print("1) Number of Word Tokens per Genre:")
print("------------------")

print("a) With stopwords")
for genre in brown.categories():
    tokens = brown.words(categories=genre)
    print(genre + ": " + str(len(tokens)))
print('\n')
"""Note that the words() function uses a different tokenizer than the word_tokenize()
    function, and will return different results"""

print("b) Without stopwords")
for genre in brown.categories():
    no_stopwords_tokens = []
    for word in brown.words(categories=genre):
        if word not in stop_words:
            no_stopwords_tokens.append(word)
    print(genre + ": " + str(len(no_stopwords_tokens)))
print('\n')

print("c) Without Stopwords and Lemmatization")
for genre in brown.categories():
    no_stopwords_tokens = []
    for word in brown.words(categories=genre):
        if word not in stop_words:
            no_stopwords_tokens.append(lemmatizer.lemmatize(word))
    print(genre + ": " + str(len(no_stopwords_tokens)))
print('\n')

print("d) Without stopwords and stemming")
for genre in brown.categories():
    no_stopwords_tokens = []
    for word in brown.words(categories=genre):
        if word not in stop_words:
            no_stopwords_tokens.append(stemmer.stem(word))
    print(genre + ": " + str(len(no_stopwords_tokens)))
print('\n')


""" Word Types per Genre """
print("------------------")
print("2) Number of Word Types per Genre:")
print("------------------")

print("a) With stopwords")
for genre in brown.categories():
    tokens = brown.words(categories=genre)
    print(genre + ": " + str(len(set(tokens))))
print('\n')

print("b) Without stopwords")
for genre in brown.categories():
    no_stopwords_tokens = []
    for word in brown.words(categories=genre):
        if word not in stop_words:
            no_stopwords_tokens.append(word)
    print(genre + ": " + str(len(set(no_stopwords_tokens))))
print('\n')

print("c) Without Stopwords and Lemmatization")
for genre in brown.categories():
    no_stopwords_tokens = []
    for word in brown.words(categories=genre):
        if word not in stop_words:
            no_stopwords_tokens.append(lemmatizer.lemmatize(word))
    print(genre + ": " + str(len(set(no_stopwords_tokens))))
print('\n')

print("d) Without stopwords and stemming")
for genre in brown.categories():
    no_stopwords_tokens = []
    for word in brown.words(categories=genre):
        if word not in stop_words:
            no_stopwords_tokens.append(stemmer.stem(word))
    print(genre + ": " + str(len(set(no_stopwords_tokens))))
print('\n')


""" Vocabulary of Whole Corpus """
print("------------------")
print("3) Vocabulary size of whole corpus:")
print("------------------")

print("a) With stopwords:")
print("Corpus vocabulary size = " + str(len(set(brown.words()))))
print('\n')

print("b) Without stopwords:")
no_stopwords_tokens = []
for word in brown.words():
    if word not in stop_words:
        no_stopwords_tokens.append(stemmer.stem(word))
print("Corpus vocabulary size = " + str(len(set(no_stopwords_tokens))))
print('\n')

print("c) Without Stopwords and Lemmatization:")
no_stopwords_tokens = []
for word in brown.words(categories=genre):
    if word not in stop_words:
        no_stopwords_tokens.append(lemmatizer.lemmatize(word))
print("Corpus vocabulary size = " + str(len(set(no_stopwords_tokens))))
print('\n')

print("d) Without stopwords and stemming:")
no_stopwords_tokens = []
for word in brown.words(categories=genre):
    if word not in stop_words:
        no_stopwords_tokens.append(stemmer.stem(word))
print("Corpus vocabulary size = " + str(len(set(no_stopwords_tokens))))