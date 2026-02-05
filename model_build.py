import pandas as pd
import numpy as np
import pickle

# Load your movies.csv
movies = pd.read_csv("movies.csv")

# Select important columns
movies = movies[['id', 'title', 'overview']]

# Fill missing values
movies['overview'] = movies['overview'].fillna('')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert overview into vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['overview']).toarray()

# Create similarity matrix
similarity = cosine_similarity(vectors)

# Save movies.pkl
pickle.dump(movies, open("movies.pkl", "wb"))

# Save similarity.pkl
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("Movies.pkl and similarity.pkl created successfully!")
