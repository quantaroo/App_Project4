# Movie Recommender System - Updated Version
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

# Select 100 Random Movies for Rating
random_movies = movies.sample(n=100, random_state=42)
movie_id_map = {movie_id: idx for idx, movie_id in enumerate(random_movies['movieId'].values)}

# Filter Ratings for Selected Movies Only
filtered_ratings = ratings[ratings['movieId'].isin(random_movies['movieId'])]

# Create Ratings Matrix for 100 Movies
def create_ratings_matrix(filtered_ratings, movie_id_map):
    row_indices = filtered_ratings['userId'] - 1
    col_indices = filtered_ratings['movieId'].map(movie_id_map)
    data = filtered_ratings['rating']
    return csr_matrix((data, (col_indices, row_indices)), shape=(100, filtered_ratings['userId'].max()))

ratings_matrix = create_ratings_matrix(filtered_ratings, movie_id_map)

# User Rating Input Function
def create_user_vector(selected_movies, movie_id_map):
    user_ratings = np.full((100,), np.nan)
    for movie_id, rating in selected_movies.items():
        if movie_id in movie_id_map:
            user_ratings[movie_id_map[movie_id]] = rating
    return csr_matrix(user_ratings.reshape(1, -1))

# Recommendation Function
def recommend_movies(user_vector, ratings_matrix, movies, top_n=10):
    if user_vector.shape[1] != ratings_matrix.shape[0]:
        st.error("Dimensional mismatch detected!")
        return pd.DataFrame(columns=["title", "genres", "Score"])
    
    similarity = cosine_similarity(user_vector, ratings_matrix)[0]
    top_indices = np.argsort(similarity)[-top_n:][::-1]
    recommendations = movies.iloc[top_indices][["title", "genres"]]
    recommendations["Score"] = similarity[top_indices]
    return recommendations

# Streamlit App
st.title("Movie Recommender System")
st.sidebar.header("Rate Movies")

# Collect User Ratings
selected_movies = {}
for idx, row in random_movies.iterrows():
    rating = st.sidebar.slider(f"{row['title']} ({row['genres']})", 1, 5, 3)
    selected_movies[row['movieId']] = rating

# Generate Recommendations
if st.sidebar.button("Show Recommendations"):
    user_vector = create_user_vector(selected_movies, movie_id_map)
    recommendations = recommend_movies(user_vector, ratings_matrix, random_movies)
    st.header("Top 10 Recommendations")
    st.table(recommendations)
