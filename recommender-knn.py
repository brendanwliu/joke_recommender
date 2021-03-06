import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from time import sleep
import random

print('A Simple KNN Recommender!\nDeveloped by: Brendan, Yerem, Josh and Elaine!')
print('----------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------')
sleep(1.5)

##Read in data
print('Reading in Data...')
input_path = '../joke_recommender/data/'
df = pd.read_csv(input_path + 'scaled_df.csv').drop(['Unnamed: 0'], axis = 1)
df_sparse = csr_matrix(df)
joketext = pd.read_csv(input_path + 'JokeText.csv')
print('Done!')

new_user = np.zeros(100)

print('Fitting Model')
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors = 100)
model_knn.fit(df_sparse.T)
print('Done!')

# Cold Start, recommend 2 jokes at random to initialize the user.
cold_read = 2
for i in range(cold_read):
    idx = random.randint(0,99)
    while(new_user[idx]!=0):
        idx = random.randint(0,99)
    init_joke = joketext.JokeText[idx]
    print(init_joke)
    print('----------------------------------------------------------------------------------')
    com = input('Rate this from -10 to 10:')
    com = float(com)/20 + 1.5
    print('----------------------------------------------------------------------------------')
    new_user[idx] = com


while True:
    if sum(new_user != 0) == 100:
        print('No More Jokes!')
        break
    distances, indices = model_knn.kneighbors(new_user.reshape(1,-1), n_neighbors = 734)
    neighbor = np.mean(df.iloc[:,np.intersect1d(df.columns.values.astype(int),indices)], axis = 1)
    neighbor[new_user != 0] = -1
    rec = np.argmax(np.array(neighbor))
    rec_joke = joketext.JokeText[rec]
    print(rec_joke)
    print('----------------------------------------------------------------------------------')
    com = input('Rate this from -10 to 10:')
    print('----------------------------------------------------------------------------------')
    if len(com) == 0:
        break
    com = float(com)/20 + 1.5
    new_user[rec] = com
    