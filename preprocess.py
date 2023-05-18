import json
import numpy as np
from itertools import chain
from scipy.sparse import coo_matrix

# Open the JSON file as a dictionary
with open('deezer_europe/deezer_europe_features.json') as feature_file:
    users = json.load(feature_file)

# Extract artist IDs
artists = set(chain.from_iterable(users.values()))

# Create feature matrix
num_users = len(users)
num_artists = max(artists) + 1

feature_matrix = np.zeros((num_users, num_artists), dtype=int)

# 1 if the artist is liked by the user, 0 otherwise
for user, liked_artists in users.items():
    for artist in liked_artists:
        feature_matrix[int(user), int(artist)] = 1

feature_matrix_coo = coo_matrix(feature_matrix)