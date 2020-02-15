# Introduction
In this project Word2vec and GloVe is implemented from scratch. Word2vec and GloVe are used for word embedding in NLP. Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers.

This project was just for learning purpose, therefore very small dataset is used. It can be trained on large datasets too with some minor changes in the code.

# Word2vec
Word2vec stands for "word to vector".
There are two approaches to implement word2vec: 
- Common bag of words(CBOW): In this approach, the network tries to predict which word is most likely given its context.
- Skig-gram: In this approach, the network uses the target word to predict its context.
In this project Word2vec is implemented by skip-gram approach.

The cost vs iteration curve of our model
![](Word2vec/images/Cost%20vs%20Iteration.png)

# GloVe
GloVe stands for "global vectors". It condenses the whole corpus in to co-occurence matrix and train on that. 

The cost vs iteration curve of our model
![](GloVe/images/Cost%20vs%20Iteration.png)

# Reference
- Word2vec: https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa
- Glove: https://nlp.stanford.edu/pubs/glove.pdf
