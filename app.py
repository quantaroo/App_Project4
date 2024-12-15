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

# Process Data
def create_user_vector(selected_movies, movies):
    user_ratings = np.zeros(len(movies))
    for movie_id, rating in selected_movies.items():
        idx = movies[movies['movieId'] == int(movie_id)].index[0]
        user_ratings[idx] = rating
    return csr_matrix(user_ratings.reshape(1, -1))

def recommend_movies(user_vector, ratings_matrix, movies, top_n=10):
    similarity = cosine_similarity(user_vector, ratings_matrix)[0]
    top_indices = similarity.argsort()[-top_n:][::-1]
    recommendations = movies.iloc[top_indices][["title", "genres"]].copy()
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
    user_vector = create_user_vector(selected_movies, movies)
    
    # Create Ratings Matrix
    ratings_matrix = csr_matrix((ratings['rating'], 
                                (ratings['movieId'] - 1, ratings['userId'] - 1)))
    
    # Generate Recommendations
    recommendations = recommend_movies(user_vector, ratings_matrix, movies)
    
    st.header("Top 10 Recommendations")
    st.table(recommendations)
