import streamlit as st
import pandas as pd
import numpy as np
import os

# Must call set_page_config before any other Streamlit commands
st.set_page_config(page_title="Movie Recommender System", layout="wide")

st.title("Movie Recommender System")

# Instructions or a brief intro at the top
st.markdown("""
**Welcome to our Movie Recommender!**  
Rate a few movies on the left, then click **Show Recommendations** to see what we suggest next.
""")

# ==============================================
# Data Loading with Error Handling
# ==============================================
@st.cache_data
def load_data():
    # Adjust these filenames/path as needed
    movies_file = "movies.csv"
    ratings_file = "ratings.csv"

    # Check if files exist
    if not os.path.exists(movies_file) or not os.path.exists(ratings_file):
        st.error("Required data files are missing. Please upload or provide 'movies.csv' and 'ratings.csv'.")
        st.stop()
        
    try:
        movies = pd.read_csv(movies_file)
        ratings = pd.read_csv(ratings_file)
    except pd.errors.EmptyDataError:
        st.error("One of the CSV files is empty. Please provide valid data.")
        st.stop()
    except pd.errors.ParserError:
        st.error("Error parsing the CSV files. Ensure 'movies.csv' and 'ratings.csv' are correctly formatted.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
        st.stop()

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
# For now, we provide dummy predictions.
# ==============================================
def myIBCF(newuser_vector, S, R):
    # Replace this with your actual IBCF logic.
    # 'newuser_vector' is an array of user's ratings for each movie (NaN if not rated).
    # 'S' is the similarity matrix, 'R' is the rating matrix.
    # Currently, we return random predictions.
    predictions = np.random.rand(len(movies)) * 5
    return predictions

# Dummy S and R placeholders (You need real computations or pre-loaded data)
S = None
R = None

# ==============================================
# Helper function to display recommendations
# ==============================================
def display_recommendations(recommendations):
    # Displays movie recommendations in a simple styled container.
    # If you have poster URLs, you can integrate st.image calls here.
    for _, row in recommendations.iterrows():
        st.markdown(f"""
        <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
            <h4 style="margin-bottom: 5px;">{row['title']}</h4>
            <p style="margin-bottom: 0;">Genres: {row['genres']}</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================
# Sidebar Configuration
# ==============================================
st.sidebar.header("Your Ratings")
system_choice = st.sidebar.radio("Choose Recommendation System:", ["System I - Popularity", "System II - IBCF"])

# Select a sample of movies to rate
if len(movies) == 0:
    st.error("No movies found. Please provide a valid movies dataset.")
    st.stop()

sample_movies = movies.sample(min(10, len(movies)), random_state=42)
selected_movies = {}
st.sidebar.write("**Rate the following movies:**")
for idx, row in sample_movies.iterrows():
    # You can adjust the default and range as needed
    rating = st.sidebar.slider(f"{row['title']}", 1, 5, 3)
    selected_movies[row['movieId']] = rating

# ==============================================
# Show Recommendations Button
# ==============================================
if st.sidebar.button("Show Recommendations"):
    # Create a user vector (dummy here)
    # In reality, map movieIds to indices using a movie_id_map and fill the vector.
    newuser_vector = np.full(len(movies), np.nan)
    # Example if you had a movie_id_map:
    # for m_id, r_val in selected_movies.items():
    #     if m_id in movie_id_map:
    #         newuser_vector[movie_id_map[m_id]] = r_val

    if system_choice == "System I - Popularity":
        st.subheader("Top 10 Popular Movies")
        recs = get_top_10_popular(popularity_df)
        if len(recs) == 0:
            st.write("No popular movies found.")
        else:
            display_recommendations(recs)

    else:
        # System II - IBCF
        if S is None or R is None:
            # If IBCF is not fully implemented
            st.warning("IBCF recommendations are not fully implemented. Showing popular recommendations instead.")
            recs = get_top_10_popular(popularity_df)
            display_recommendations(recs)
        else:
            # When you have the actual S and R matrices, call the real IBCF function here.
            preds = myIBCF(newuser_vector, S, R)
            # Sort by predicted rating
            non_na_preds = np.where(~np.isnan(preds))[0]
            if len(non_na_preds) > 0:
                top_pred_indices = non_na_preds[np.argsort(preds[non_na_preds])][-10:][::-1]
                recommended_movies = movies.iloc[top_pred_indices].copy()
                recommended_movies['predicted_rating'] = preds[top_pred_indices]

                st.subheader("Top 10 IBCF Recommendations")
                cols = st.columns(2)
                for i, rec_row in recommended_movies.iterrows():
                    with cols[i % 2]:
                        # If you have poster URLs:
                        # st.image(rec_row.get('poster_url', ''), width=150)
                        st.write(f"**{rec_row['title']}**")
                        st.write(f"Genres: {rec_row['genres']}")
                        st.write(f"Predicted Rating: {rec_row['predicted_rating']:.2f}")
                        st.write("---")
            else:
                # If no predictions, fallback to popular
                st.info("Not enough data for IBCF predictions. Showing popular recommendations instead.")
                fallback = get_top_10_popular(popularity_df)
                display_recommendations(fallback)
else:
    st.info("Please rate some movies on the left and then click **Show Recommendations**.")
