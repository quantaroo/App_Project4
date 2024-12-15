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

# Select Random Movies for the Interface
random_movies = movies.sample(100, random_state=42).reset_index(drop=True)
movie_id_map = {mid: idx for idx, mid in enumerate(random_movies['movieId'])}

# Create Ratings Matrix
@st.cache_data
def create_ratings_matrix(ratings, movie_id_map):
    valid_indices = ratings['movieId'].isin(movie_id_map.keys())
    filtered_ratings = ratings.loc[valid_indices]

    row_indices = filtered_ratings['movieId'].map(movie_id_map).to_numpy()
    col_indices = (filtered_ratings['userId'] - 1).to_numpy()
    data = filtered_ratings['rating'].to_numpy()

    num_users = ratings['userId'].max()
    ratings_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(100, num_users))
    return ratings_matrix

ratings_matrix = create_ratings_matrix(ratings, movie_id_map)

# Create User Vector
@st.cache_data
def create_user_vector(selected_movies, movie_id_map):
    user_ratings = np.zeros(len(movie_id_map))
    for movie_id, rating in selected_movies.items():
        if movie_id in movie_id_map:
            user_ratings[movie_id_map[movie_id]] = rating
    return csr_matrix(user_ratings.reshape(1, -1))

# Recommend Movies
@st.cache_data
def recommend_movies(user_vector, ratings_matrix, movies):
    if user_vector.shape[1] != ratings_matrix.shape[0]:
        st.error("Dimensional mismatch detected!")
        return pd.DataFrame(columns=["title", "genres", "Score"])

    similarity = cosine_similarity(user_vector, ratings_matrix)[0]
    top_indices = np.argsort(similarity)[-10:][::-1]
    recommendations = movies.iloc[top_indices][["title", "genres"]]
    recommendations["Score"] = similarity[top_indices]
    return recommendations

# Streamlit App Interface
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
