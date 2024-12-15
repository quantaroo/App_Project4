# Streamlit Movie Recommender System

import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# Filter Ratings by Available Movies
filtered_ratings = ratings[ratings['movieId'].isin(movies['movieId'])]

# Re-index Movie IDs
movie_id_map = {movie_id: idx for idx, movie_id in enumerate(movies['movieId'].unique())}

# Create Ratings Matrix
ratings_matrix = csr_matrix(
    (filtered_ratings['rating'], 
     (filtered_ratings['movieId'].map(movie_id_map), filtered_ratings['userId']))
)

# Create User Vector
def create_user_vector(selected_movies):
    user_ratings = np.zeros(movies.shape[0])
    for movie_id, rating in selected_movies.items():
        if movie_id in movie_id_map:
            user_ratings[movie_id_map[movie_id]] = rating
    return csr_matrix(user_ratings.reshape(1, -1))

# Recommend Movies
def recommend_movies(user_vector, ratings_matrix, movies, top_n=10):
    st.write(f"DEBUG - User Vector Shape: {user_vector.shape}")
    st.write(f"DEBUG - Ratings Matrix Shape: {ratings_matrix.shape}")
    if user_vector.shape[1] != ratings_matrix.shape[0]:
        st.error("User vector and ratings matrix dimensions do not match.")
        return pd.DataFrame(columns=["title", "genres", "Score"])

    similarity = cosine_similarity(user_vector, ratings_matrix)[0]
    top_indices = np.argsort(similarity)[-top_n:][::-1]
    recommendations = movies.iloc[top_indices][["title", "genres"]]
    recommendations["Score"] = similarity[top_indices]
    return recommendations

# Streamlit App
st.title("Movie Recommender System")
st.sidebar.header("Rate Movies")

# User Ratings Input
selected_movies = {}
for idx, row in movies.iterrows():
    rating = st.sidebar.slider(f"{row['title']} ({row['genres']})", 1, 5, 3)
    selected_movies[row['movieId']] = rating

# Recommendation Button
if st.sidebar.button("Show Recommendations"):
    user_vector = create_user_vector(selected_movies)
    recommendations = recommend_movies(user_vector, ratings_matrix, movies)

    st.header("Top 10 Recommendations")
    st.table(recommendations)
