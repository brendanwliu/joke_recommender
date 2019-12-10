import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

##############
# Based on an input of one joke, getting k nearest neighbors of people who found that joke funny
# recommending jokes that the neighbors also found funny.
##############



##Read in data
print('Reading in Joke Data...')
input_path = '../joke_recommender/data/'
df = pd.read_csv(input_path + 'scaled_df.csv').drop(['Unnamed: 0'], axis = 1)
df_sparse = csr_matrix(df)
joketext = pd.read_csv(input_path + 'JokeText.csv')
print('Done!')

print('Fitting knn-Model')
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors = 100)
model_knn.fit(df_sparse)
print('Done!')

while True:
    com = input('Which Joke did you like? \nJoke number:')
    if len(com) == 0:
        break
    query_index = int(com) + 1
    distances, indices = model_knn.kneighbors(df.iloc[query_index,:].values.reshape(1,-1), n_neighbors = 6)
    for i in range(0,len(distances.flatten())):
        if i == 0:
            print('Top 5 recommendations of joke # {0}:\n'.format(query_index))
        else:
            print('{0}: {1}\nwith distance {2}\n'.format(i, joketext.JokeText.iloc[indices.flatten()[i]], distances.flatten()[i]))