from collections import defaultdict
import numpy as np

class Data():

    def __init__(self, setting, corpus):

        self.window = setting.window
        self.corpus = corpus

    def prepare_training_data(self):
        """
        It prepares the data for training.
        """

        self.words_freq = defaultdict(int)

        # Counting Frequency of each word
        for sentence in self.corpus:
            for word in sentence:
                self.words_freq[word] += 1

        self.words = list(self.words_freq.keys())                                # List of unique words
        self.words_num = len(self.words)                                         # Number of unique words
        self.word_index = dict([[word, i] for i, word in enumerate(self.words)]) # It is the dictionary whose keys are word and values are their indices, generated randomly.
        self.index_word = dict([[i, word] for i, word in enumerate(self.words)]) # It is the dictionary whose keys are indices of words and values are words corresponding to that index.

        training_data = [] # It will the list of target word (vector) with its context words (vectors).

        for sentence in self.corpus:

            sen_len = len(sentence)

            for j, word in enumerate(sentence):

                target = self.word2onehotvec(word)
                context = [0 for _ in range(self.words_num)]

                for k in range(j-self.window, j+self.window+1):

                    if k != j and 0 <= k < sen_len:

                        context_word = sentence[k]
                        index = self.word_index[context_word]
                        context[index] += 1

                training_data.append([target, context])

        self.training_data = np.array(training_data)
        self.m, __, ___ = self.training_data.shape

    def word2onehotvec(self, word):

        # Creating a zero vector of length self.words_num
        onehotvec = [0 for _ in range(self.words_num)]
        index = self.word_index[word]
        onehotvec[index] = 1
        return onehotvec


