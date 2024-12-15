import streamlit as st
import pandas as pd
import numpy as np

# set_page_config must come before any UI commands
st.set_page_config(page_title="Movie Recommender System", layout="wide")

# Now proceed with the rest of your code
st.title("Movie Recommender System")

# ==============================================
# Data Loading (Dummy Placeholders)
# ==============================================
@st.cache_data
def load_data():
    # Replace with your actual data loading
    movies = pd.read_csv("movies.csv")
    # Example: movies DataFrame with columns ['movieId', 'title', 'genres', 'poster_url']
    # Add 'poster_url' if you have it. If not, omit the image display part.
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# ==============================================
# System I: Popularity-Based Recommendations
# ==============================================
@st.cache_data
def compute_popularity(movies, ratings, min_count=50):
    movie_counts = ratings.groupby('movieId')['rating'].count()
    movie_means = ratings.groupby('movieId')['rating'].mean()
    popularity_df = pd.DataFrame({'count': movie_counts, 'mean_rating': movie_means})
    popularity_df = popularity_df[popularity_df['count'] >= min_count].copy()
    popularity_df['popularity_score'] = popularity_df['mean_rating'] * np.log(popularity_df['count'])
    popularity_df = popularity_df.merge(movies, on='movieId', how='left')
    popularity_df = popularity_df.sort_values('popularity_score', ascending=False)
    return popularity_df

popularity_df = compute_popularity(movies, ratings)

def get_top_10_popular(popularity_df):
    return popularity_df[['movieId', 'title', 'genres']].head(10)

# ==============================================
# Placeholder for IBCF (You should integrate your logic here)
# ==============================================
def myIBCF(newuser_vector, S, R):
    # Your IBCF prediction logic here
    # Return a vector of predictions for all movies
    predictions = np.random.rand(len(movies)) * 5  # Dummy predictions
    return predictions

# Dummy S and R placeholders (You need real computations or pre-loaded data)
S = None
R = None

# ==============================================
# Layout and UI to mimic the look
# ==============================================

st.set_page_config(page_title="Movie Recommender System", layout="wide")
st.title("Movie Recommender System")

# Instructions or a brief intro at the top
st.markdown("""
**Welcome to our Movie Recommender!**  
Rate a few movies on the left, then click **Show Recommendations** to see what we suggest next.
""")

# Sidebar: choose system and movies to rate
st.sidebar.header("Your Ratings")
system_choice = st.sidebar.radio("Choose Recommendation System:", ["System I - Popularity", "System II - IBCF"])

# For a neat look, select a handful of movies to rate
sample_movies = movies.sample(10, random_state=42)
selected_movies = {}
st.sidebar.write("**Rate the following movies:**")
for idx, row in sample_movies.iterrows():
    # You can adjust the default and range as needed
    rating = st.sidebar.slider(f"{row['title']}", 1, 5, 3)
    selected_movies[row['movieId']] = rating

if st.sidebar.button("Show Recommendations"):
    # Create a user vector (dummy here)
    # In reality, map movieIds to indices and fill vector
    newuser_vector = np.full(len(movies), np.nan)
    # If you have movie_id_map, apply it here:
    # for m_id, r_val in selected_movies.items():
    #     if m_id in movie_id_map:
    #         newuser_vector[movie_id_map[m_id]] = r_val

    # =====================================================
    # Display Recommendations
    # =====================================================
    if system_choice == "System I - Popularity":
        st.subheader("Top 10 Popular Movies")
        recs = get_top_10_popular(popularity_df)
        for _, row in recs.iterrows():
            st.write(f"**{row['title']}** ({row['genres']})")
    else:
        # System II - IBCF
        # In reality, call myIBCF with real S, R, etc.
        preds = myIBCF(newuser_vector, S, R)
        # Sort by predicted rating
        non_na_preds = np.where(~np.isnan(preds))[0]
        if len(non_na_preds) > 0:
            top_pred_indices = non_na_preds[np.argsort(preds[non_na_preds])][-10:][::-1]
            recommended_movies = movies.iloc[top_pred_indices].copy()
            recommended_movies['predicted_rating'] = preds[top_pred_indices]
            
            st.subheader("Top 10 IBCF Recommendations")
            # Display recommendations in a neat layout
            # For a grid-like display of posters and titles:
            cols = st.columns(2)
            for i, row in recommended_movies.iterrows():
                with cols[i % 2]:
                    # If you have poster URLs:
                    # st.image(row.get('poster_url', ''), width=150)
                    st.write(f"**{row['title']}**")
                    st.write(f"Genres: {row['genres']}")
                    st.write(f"Predicted Rating: {row['predicted_rating']:.2f}")
                    st.write("---")
        else:
            # If no predictions, fallback to popular
            st.subheader("Not enough data for IBCF predictions. Showing Popular Recommendations:")
            fallback = get_top_10_popular(popularity_df)
            for _, row in fallback.iterrows():
                st.write(f"**{row['title']}** ({row['genres']})")

else:
    st.info("Please rate some movies on the left and then click **Show Recommendations**.")
