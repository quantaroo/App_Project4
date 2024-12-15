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

# Process Data
def create_user_vector(selected_movies):
    user_ratings = np.zeros(movies.shape[0])
    for movie_id, rating in selected_movies.items():
        user_ratings[int(movie_id) - 1] = rating
    return csr_matrix(user_ratings.reshape(1, -1))

def recommend_movies(user_vector, ratings_matrix, movies, top_n=10):
    similarity = cosine_similarity(user_vector, ratings_matrix)[0]
    top_indices = similarity.argsort()[-top_n:][::-1]
    recommendations = movies.iloc[top_indices][["title", "genres"]]
    recommendations["Score"] = similarity[top_indices]
    return recommendations

# Build Streamlit App
st.title("Movie Recommender System")
st.sidebar.header("Rate Movies")

# User Ratings Input
selected_movies = {}
for index, row in movies.iterrows():
    rating = st.sidebar.slider(f"{row['title']} ({row['genres']})", 1, 5, 3)
    selected_movies[row['movieId']] = rating

# Recommendation Button
if st.sidebar.button("Show Recommendations"):
    user_vector = create_user_vector(selected_movies)
    ratings_matrix = csr_matrix((ratings['rating'], (ratings['movieId']-1, ratings['userId']-1)))
    recommendations = recommend_movies(user_vector, ratings_matrix, movies)
    
    st.header("Top 10 Recommendations")
    st.table(recommendations)

