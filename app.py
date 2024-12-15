import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# Data Loading and Preprocessing
# =========================================================

@st.cache_data
def load_data():
    # Adjust file paths as necessary
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# =========================================================
# System I: Popularity-Based Recommendation
# =========================================================

@st.cache_data
def compute_popularity(movies, ratings, min_count=50):
    # Define popularity: for example, movies with at least 'min_count' ratings,
    # and ranking them by average rating * log(number_of_ratings).
    movie_counts = ratings.groupby('movieId')['rating'].count()
    movie_means = ratings.groupby('movieId')['rating'].mean()
    popularity_df = pd.DataFrame({'count': movie_counts, 'mean_rating': movie_means})
    # Filter by minimum number of ratings
    popularity_df = popularity_df[popularity_df['count'] >= min_count].copy()
    # Define a popularity score (this is arbitrary; adapt as needed)
    popularity_df['popularity_score'] = popularity_df['mean_rating'] * np.log(popularity_df['count'])
    # Merge with movie titles
    popularity_df = popularity_df.merge(movies, on='movieId', how='left')
    return popularity_df.sort_values('popularity_score', ascending=False)

popularity_df = compute_popularity(movies, ratings)

def get_top_10_popular(popularity_df):
    return popularity_df[['movieId', 'title', 'genres']].head(10)

# =========================================================
# System II: IBCF Setup
# =========================================================

# Filter ratings to ensure consistency
filtered_ratings = ratings[ratings['movieId'].isin(movies['movieId'])]
movie_id_map = {movie_id: idx for idx, movie_id in enumerate(movies['movieId'].unique())}
user_id_map = {uid: idx for idx, uid in enumerate(filtered_ratings['userId'].unique())}

num_movies = len(movie_id_map)
num_users = len(user_id_map)

# Create user-item matrix (Note: Many entries will be NA, we store as sparse)
R = np.full((num_users, num_movies), np.nan)
for row in filtered_ratings.itertuples():
    u = user_id_map[row.userId]
    m = movie_id_map[row.movieId]
    R[u, m] = row.rating

# ---------------------------------------------------------
# Normalizing the rating matrix by subtracting row means
# Only subtract mean of non-NA values
row_means = np.nanmean(R, axis=1)
R_centered = R - row_means[:, np.newaxis]
# Keep NaNs where R was NaN
R_centered[np.isnan(R)] = np.nan

# ---------------------------------------------------------
# Compute Item-Item Similarity:
# According to instructions, only consider pairs with > 2 overlapping users.
# Then cos(i,j) = sum over l in I_ij of R_centered[l,i]*R_centered[l,j] / sqrt(...)
# Then transform similarity = (1 + cos)/2

@st.cache_data
def compute_item_similarity(R_centered, min_common=3):
    # R_centered: num_users x num_movies
    # We'll compute similarity for all pairs; this can be large (3706x3706).
    # For demonstration, we’ll do a simplified loop. In practice, optimize or precompute offline.
    
    num_movies = R_centered.shape[1]
    S = np.full((num_movies, num_movies), np.nan)
    
    # Compute norms
    norms = np.nansum(R_centered**2, axis=0)**0.5
    
    for i in range(num_movies):
        for j in range(i+1, num_movies):
            # Find users who rated both i and j
            mask = ~np.isnan(R_centered[:, i]) & ~np.isnan(R_centered[:, j])
            common_users = np.where(mask)[0]
            if len(common_users) > min_common:
                # Compute cosine similarity
                dot_ij = np.nansum(R_centered[common_users, i] * R_centered[common_users, j])
                denom = norms[i]*norms[j]
                if denom > 0:
                    cos_ij = dot_ij / denom
                    sim_ij = (1 + cos_ij)/2.0
                    S[i, j] = sim_ij
                    S[j, i] = sim_ij
    return S

S = compute_item_similarity(R_centered, min_common=2)

# ---------------------------------------------------------
# For each movie, keep top 30 neighbors
@st.cache_data
def keep_top_30(S):
    # For each row (movie), keep top 30 sims and set others to NaN
    num_movies = S.shape[0]
    for i in range(num_movies):
        row = S[i, :]
        # Find top 30 indices
        valid_indices = np.where(~np.isnan(row))[0]
        if len(valid_indices) > 30:
            top_30 = valid_indices[np.argsort(row[valid_indices])[-30:]]
            # Set others to NaN
            mask = np.ones_like(row, dtype=bool)
            mask[top_30] = False
            row[mask] = np.nan
            S[i, :] = row
    return S

S = keep_top_30(S)

# =========================================================
# IBCF Prediction Function (myIBCF)
# =========================================================

def myIBCF(newuser_vector, S, R):
    # newuser_vector (w): 3706-by-1 vector (in this case, num_movies-by-1)
    # R is original rating matrix (not centered), or you can use w directly.
    # According to the formula:
    # pred(i) = ( sum_{j in S(i), w_j != NA} S_ij * w_j ) / ( sum_{j in S(i), w_j != NA} S_ij )
    
    predictions = np.full(S.shape[0], np.nan)
    for i in range(S.shape[0]):
        neighbors = np.where(~np.isnan(S[i, :]))[0]
        # Only consider neighbors user has rated
        rated_neighbors = [j for j in neighbors if not np.isnan(newuser_vector[j])]
        if len(rated_neighbors) > 0:
            numer = np.nansum(S[i, rated_neighbors] * newuser_vector[rated_neighbors])
            denom = np.nansum(S[i, rated_neighbors])
            if denom != 0:
                predictions[i] = numer / denom
    return predictions

# =========================================================
# Streamlit App UI
# =========================================================

st.title("Movie Recommender System")

system_choice = st.radio("Choose Recommendation System:", ["System I - Popularity", "System II - IBCF"])

if system_choice == "System I - Popularity":
    st.header("Top 10 Popular Movies")
    top10 = get_top_10_popular(popularity_df)
    st.table(top10)

elif system_choice == "System II - IBCF":
    st.header("Rate Some Movies")

    # Limit the number of movies displayed to 20 for practicality
    sample_movies = movies.sample(20, random_state=42)
    selected_movies = {}
    for idx, row in sample_movies.iterrows():
        rating = st.slider(f"{row['title']} ({row['genres']})", 1, 5, 3)
        selected_movies[row['movieId']] = rating

    if st.button("Show IBCF Recommendations"):
        # Create new user vector
        newuser_vector = np.full(num_movies, np.nan)
        for m_id, r_val in selected_movies.items():
            if m_id in movie_id_map:
                newuser_vector[movie_id_map[m_id]] = r_val
        
        # Use IBCF to predict for unrated items
        preds = myIBCF(newuser_vector, S, R)
        
        # If fewer than 10 predictions are non-NA, fill with popular movies
        non_na_preds = np.where(~np.isnan(preds))[0]
        if len(non_na_preds) < 10:
            # Get already rated by user to exclude
            rated_by_user = [movie_id_map[m] for m in selected_movies.keys() if m in movie_id_map]
            # Fill the gap with popular movies not rated by the user
            shortage = 10 - len(non_na_preds)
            additional = popularity_df[~popularity_df['movieId'].isin(selected_movies.keys())].head(shortage)
            # Combine results: first the predicted non-NA, sorted by prediction
            if len(non_na_preds) > 0:
                top_pred_indices = non_na_preds[np.argsort(preds[non_na_preds])][-len(non_na_preds):][::-1]
                top_pred_movies = movies.iloc[top_pred_indices]
                top_pred_movies['prediction'] = preds[top_pred_indices]
                recommendations = pd.concat([
                    top_pred_movies[['movieId', 'title', 'genres', 'prediction']],
                    additional[['movieId', 'title', 'genres']].assign(prediction=np.nan)
                ], ignore_index=True)
                recommendations = recommendations.head(10)
            else:
                # No predictions at all: just show popularity
                recommendations = additional[['movieId', 'title', 'genres']].head(10)
        else:
            # We have enough predictions
            top_pred_indices = non_na_preds[np.argsort(preds[non_na_preds])][-10:][::-1]
            recommendations = movies.iloc[top_pred_indices].copy()
            recommendations['prediction'] = preds[top_pred_indices]

        st.header("Top 10 IBCF Recommendations")
        st.table(recommendations[['title', 'genres', 'prediction']])
