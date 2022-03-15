import nltk
import random
from nltk.corpus import brown
from nltk.lm import MLE
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('brown')

"""
sentence_gen function
This function converts a model's returned set
    (with given length and random seed) into a
    detokenized sentence.
    
Parameters:
    model: The language model being used
    sentence_length: Number of words in sentence
    rand_seed: The seed for the random number
"""


def sentence_gen(model, sentence_length, rand_seed=1):
    sentence = []
    for token in model.generate(sentence_length, random_seed=rand_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        sentence.append(token)
    return TreebankWordDetokenizer().detokenize(sentence)


"""
 Question 3 a) sentences with bigrams
 """
padded_line = [list(pad_both_ends(brown.words(), n=2))]
train, vocab = padded_everygram_pipeline(2, padded_line)

lmodel = MLE(2)
lmodel.fit(train, vocab)

print("Question 3 a) Random sentence using bigrams:")
print(sentence_gen(lmodel, 10, rand_seed=random.randint(1, len(brown.words()))))
print("\n")


"""
 Question 3 b) sentences with trigrams
 """
print("Question 3 b) Random sentence using trigrams:")
padded_line = [list(pad_both_ends(brown.words(), n=3))]
train, vocab = padded_everygram_pipeline(3, padded_line)

lmodel = MLE(3)
lmodel.fit(train, vocab)

print(sentence_gen(lmodel, 10, rand_seed=random.randint(1, len(brown.words()))))
print("\n")


