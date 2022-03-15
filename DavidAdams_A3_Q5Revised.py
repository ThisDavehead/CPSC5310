import gensim
from gensim import corpora, models, similarities
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec
import numpy as np
import nltk
import random
from nltk.corpus import brown, stopwords
import string
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api
from gensim.scripts.glove2word2vec import glove2word2vec

nltk.download('brown')
nltk.download('stopwords')

# convert data into a list of 15 tokenized strings (one for each category)
lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')
cat_word_set = [
    [lemmatizer.lemmatize(word.lower()) for word in brown.words(categories=genre) if word not in stopwords
     and word not in string.punctuation and len(lemmatizer.lemmatize(word)) >= 3]
    for genre in brown.categories()]

print("Q5a) TF-IDF Similarity")
# Part i: Compute similarity between categories
print("Calculating similarity between all categories")
dictionary = corpora.Dictionary(cat_word_set)
corpus = [dictionary.doc2bow(text) for text in cat_word_set]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# Now calculate similarity matrix
index = similarities.MatrixSimilarity(corpus_tfidf)
sim_matrix = index[corpus_tfidf]
np.set_printoptions(precision=3)
print("Matrix of similarities between clusters:")
for i in range(0, 15):
    print(str(i) + ") " + brown.categories()[i] + ": ")
    print(sim_matrix[i])
print("")


# Part ii: Compute similarity within categories
print("Calculating similarity within all categories")
for i in range(0, 15):
    # Split the category's text in half, then rejoin them into one list of texts
    mid_point = len(cat_word_set[i])//2
    left_text = cat_word_set[i][:mid_point]
    right_text = cat_word_set[i][mid_point:]
    full_text = [left_text, right_text]

    # Create corpus and dictionary from list of text halves
    dictionary = corpora.Dictionary(full_text)
    full_corpus = [dictionary.doc2bow(text) for text in full_text]

    tfidf = models.TfidfModel(full_corpus)
    corpus_tfidf = tfidf[full_corpus]

    # Now create the similarity matrix
    index = similarities.MatrixSimilarity(tfidf[full_corpus])
    cat_sim_matrix = index[corpus_tfidf]
    print(brown.categories()[i] + ": ", cat_sim_matrix[0][1])


print(" ")
# Word2Vec Part (b)
print("Q5b) Word2Vec Cosine Similarity")
print("Calculating similarity between all categories")
# Part i: Compute similarity within categories
dictionary = corpora.Dictionary(cat_word_set)
corpus = [dictionary.doc2bow(text) for text in cat_word_set]

print("downloading pre-trained model...")
glove = api.load("glove-wiki-gigaword-100")
print("Generating term index...")
term_index = models.keyedvectors.WordEmbeddingSimilarityIndex(glove)
print("Generating similarity matrix...")
similarity_matrix = similarities.SparseTermSimilarityMatrix(term_index, dictionary)
print("Generating cosine similarity index...")
doc_index = similarities.SoftCosineSimilarity(corpus, similarity_matrix)

# Now calculate similarity matrix
np.set_printoptions(precision=3)
print("Matrix of similarities between clusters:")
for i in range(0, 15):
    print(str(i) + ") " + brown.categories()[i] + ": ")
    print(doc_index[dictionary.doc2bow(cat_word_set[i])])
print("")


# Part ii: Compute similarity within categories
print("Calculating similarity within all categories")
for i in range(0, 15):
    # Split the category's text in half, then rejoin them into one list of texts
    mid_point = len(cat_word_set[i])//2
    left_text = cat_word_set[i][:mid_point]
    right_text = cat_word_set[i][mid_point:]
    full_text = [left_text, right_text]

    # Create corpus and dictionary from list of text halves
    new_dictionary = corpora.Dictionary(full_text)
    full_corpus = [new_dictionary.doc2bow(text) for text in full_text]

    # Now create the similarity matrix
    doc_index = similarities.SoftCosineSimilarity(full_corpus, similarity_matrix)

    np.set_printoptions(precision=3)
    sims = doc_index[new_dictionary.doc2bow(left_text)]
    print(brown.categories()[i] + ": ", sims[1])


