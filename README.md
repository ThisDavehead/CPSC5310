# CPSC5310
Assignments from course in Natural Language Processing in Spring of 2020.

**Questions Answered:**  
**A1**  
Given the Brown corpus available from NLTK (Natural Language Processing Tool Kit),  
it consists of several categories/genres of texts, write a program that computes:  
* How many word tokens does each category/genre have?  
* How many word types does each category/genre have?  
* What is the vocabulary size of the whole corpus?  

Your program should compute with the following variations:  
(a) with stopwords,  
(b) without stopwords,  
(c) without stopwords and lemmatization, and  
(d) without stopwords and stemming.  
You may display them by category. On our lab computers, you have to run python3.6 
because if you just type in python it defaults to python2.7 and NLTK is not available in
that version.  
  
**DavidAdams_A2_Q3**  
Using a language model generated from the Brown corpus, write a program that generates
random sentences (i.e., Shannon Visualization Method) using:  
(a) bi-grams  
(b) tri-grams  
  
**David_Adams_A2_Q4**  
Using the Brown corpus, write a program that classify texts according to one of the categories/genres, and evaluate your program. You should use:  
(a) Naive Bayes  
(b) Logistic regression  
You should split the data into training and testing (e.g., 70% and 30%), mix up the testing
data and see if your classifier will succeed to split them into categories/genres again.  
  
**DavidAdams_A3_Q5Revised**  
Given 2 text documents, write a function that computes the similarity between the 2
documents using the cosine similarity measure with:  
(a) TF-IDF representation  
(b) Word2Vec representation  
Apply your function to the Brown corpus to compute the similarity within the cluster and
between clusters.  
  
**DavidAdams_A4**  
Using the Brown corpus, write a program that classify texts according to one of the categories/genres, and evaluate the results of your program. You should use Word2Vec to
represent the words, and the two following classifiers:  
(a) Logistic regression or single-layer perceptron, and  
(b) Multi-layer neural network (about 3 or 5 layers with 2 or 3 units each).  
You should split the data into training and testing (e.g., 90% and 10%), mix up the testing
data and see if your classifiers will succeed to split them into categories/genres again.  
