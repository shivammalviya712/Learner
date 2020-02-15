# Date 5th October
# Writer - Shivam Malviya
# Goal - Implementing GloVe algorithm from scratch

""" Conventions

   m - Window size
   d - Dimension of vectors
   a - Learning rate
   U - Matrix for hidden layer calculation
   V - Matrix for output calculation
   X - Co-occurrence matrix
   T - Total number of unique words
   J - Loss
   u - target vector
   v - context vector

"""

from training_data import Data
from settings import Settings
from model import Model


settings = Settings()
data = Data(settings)
model = Model(data, settings)
model.train()

word = input('Enter the word whose vector you wanna see.\n')
print(model.predict(word))



