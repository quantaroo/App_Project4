import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Select 100 Random Movies
random_movies = movies.sample(100, random_state=42)
movie_id_map = {row["movieId"]: idx for idx, row in random_movies.iterrows()}

# Function to Create Ratings Matrix
def create_ratings_matrix(ratings, movie_id_map):
    relevant_ratings = ratings[ratings['movieId'].isin(movie_id_map.keys())]
    row_indices = relevant_ratings['movieId'].map(movie_id_map).values
    col_indices = relevant_ratings['userId'] - 1
    data = relevant_ratings['rating'].values
    return csr_matrix((data, (row_indices, col_indices)), shape=(100, ratings['userId'].max()))

ratings_matrix = create_ratings_matrix(ratings, movie_id_map)

# User Rating Input Function
def create_user_vector(selected_movies, movie_id_map):
    user_ratings = np.zeros(len(movie_id_map))
    for movie_id, rating in selected_movies.items():
        if movie_id in movie_id_map:
            index = movie_id_map[movie_id]
            user_ratings[index] = rating
    return csr_matrix(user_ratings.reshape(1, -1))

# Recommendation Function
def recommend_movies(user_vector, ratings_matrix, movies, top_n=10):
    similarity = cosine_similarity(user_vector, ratings_matrix)[0]
    top_indices = np.argsort(similarity)[-top_n:][::-1]
    recommendations = movies.iloc[top_indices][["title", "genres"]]
    recommendations["Score"] = similarity[top_indices]
    return recommendations

# Build Streamlit App
st.title("Movie Recommender System")
st.sidebar.header("Rate Movies")

# Collect User Ratings
selected_movies = {}
for _, row in random_movies.iterrows():
    rating = st.sidebar.slider(f"{row['title']} ({row['genres']})", 1, 5, 3)
    selected_movies[row['movieId']] = rating

# Generate Recommendations
if st.sidebar.button("Show Recommendations"):
    user_vector = create_user_vector(selected_movies, movie_id_map)
    recommendations = recommend_movies(user_vector, ratings_matrix, random_movies)

    st.header("Top 10 Recommendations")
    st.table(recommendations)
