# Date 9th September
# Writer - Shivam Malviya
# Goal - Learning Word2vec Algorithm

import training_data as td
from settings import Settings
from model import Model


settings = Settings()

text = 'natural language processing and machine learning is fun and exciting'
corpus = [[word.lower() for word in text.split()]]

data = td.Data(settings, corpus) # Creating the object data.
data.prepare_training_data()     # Preparing Data for the purpose of training.

model = Model(data, settings)
model.train()
print('Press any button to stop.')



