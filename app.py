# Streamlit Movie Recommender System

import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load Data
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    movies = pd.read_csv(os.path.join(current_dir, "movies.csv"))
    ratings = pd.read_csv(os.path.join(current_dir, "ratings.csv"))
    return movies, ratings

movies, ratings = load_data()

# Create User Rating Vector
def create_user_vector(selected_movies, max_movie_id):
    user_ratings = np.zeros(max_movie_id)
    for movie_id, rating in selected_movies.items():
        if 0 <= movie_id - 1 < max_movie_id:
            user_ratings[movie_id - 1] = rating
    return csr_matrix(user_ratings.reshape(1, -1))

# Recommend Movies
def recommend_movies(user_vector, ratings_matrix, movies, top_n=10):
    # Ensure Correct Dimensions
    if user_vector.shape[1] != ratings_matrix.shape[0]:
        st.error("Dimensional mismatch detected! Check movie and rating data.")
        return pd.DataFrame(columns=["title", "genres", "Score"])
    
    similarity = cosine_similarity(user_vector, ratings_matrix)[0]
    top_indices = np.argsort(similarity)[-top_n:][::-1]
    recommendations = movies.iloc[top_indices][["title", "genres"]]
    recommendations["Score"] = similarity[top_indices]
    return recommendations

# Validate and Create Ratings Matrix
st.title("Movie Recommender System")
st.sidebar.header("Rate Movies")

# Validate Matrix Dimensions
max_movie_id = movies['movieId'].max()
max_user_id = ratings['userId'].max()

if ratings['movieId'].max() > max_movie_id or ratings['userId'].max() > max_user_id:
    st.error("Invalid indices in the ratings dataset. Check for mismatches.")
else:
    ratings_matrix = csr_matrix(
        (ratings['rating'], 
         (ratings['movieId'] - 1, ratings['userId'] - 1)),
        shape=(max_movie_id, max_user_id)
    )

# User Ratings Input
selected_movies = {}
for _, row in movies.iterrows():
    rating = st.sidebar.slider(f"{row['title']} ({row['genres']})", 1, 5, 3)
    selected_movies[row['movieId']] = rating

# Recommendation Button
if st.sidebar.button("Show Recommendations"):
    user_vector = create_user_vector(selected_movies, max_movie_id)
    recommendations = recommend_movies(user_vector, ratings_matrix, movies)
    
    st.header("Top 10 Recommendations")
    st.table(recommendations)
