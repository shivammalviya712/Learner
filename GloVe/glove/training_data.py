from collections import defaultdict
import numpy as np

class Data():

    def __init__(self, settings):

        self.m = settings.m
        self.make_corpus()
        self.co_occurrence_matrix()

    def make_corpus(self):

        text = "i love doing this project it is the thing which i enjoy the most"
        corpus = [[word.lower() for word in text.split()]]
        self.corpus = corpus

    def preparation(self):

        """ Making the frequency matrix"""
        self.word_freq = defaultdict(int)

        for sentence in self.corpus:
            for word in sentence:
                self.word_freq[word] += 1

        # self.words decide the index of all the words
        self.words = list(self.word_freq.keys())
        self.T = len(self.words)

        # word_index will give index for a given word and vice versa for index_word
        self.word_index = dict([[word, i] for i, word in enumerate(self.words)])
        self.index_word = dict([[i, word] for i, word in enumerate(self.words)])

    def co_occurrence_matrix(self):

        self.preparation()

        # In X each word have a particular row and column according to their index.
        # Xij - tabulate number of times word j appear in the context of word i
        self.X = np.zeros([self.T, self.T])
        self.X += 0.0001

        for sentence in self.corpus:

            sen_len = len(sentence)

            for i, c_word in enumerate(sentence):
                # target is the center word index
                # context is the context word index
                target = self.word_index[c_word]

                for j in range(i-self.m, i+self.m+1):

                    if j != i and 0 <= j and j < sen_len:

                        context_word = sentence[j]
                        context = self.word_index[context_word]
                        self.X[target, context] += 1 / abs(i-j)










